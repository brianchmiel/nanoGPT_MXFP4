// #include <torch/extension.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cassert>
#include <cstdint>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#pragma STDC FENV_ACCESS ON
#include <cfenv>


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define STOCH_ROUNDING_MIN  -2147483648 //-(2 ** 31)
#define STOCH_ROUNDING_MAX  2147483647 //2**31 - 1

#define SIGN_OFFSET_FP32      31
#define EXPONENT_OFFSET_FP32  23
#define EXPONENT_BIAS_FP32    127
#define SIGNIFICAND_MASK_FP32 0x007FFFFF
#define EXPONENT_MASK_FP32    0x7F800000
#define SIGN_MASK_FP32        0x80000000
#define SIGN_MASK_16B 0x8000
#define SIGN_MASK_8B 0x80
#define SIGN_MASK_FP4 0x8





//sbs implements select bits x[high:low]
__device__ uint32_t sbs(uint32_t x, uint8_t high, uint8_t low)
{
  return (high==31) ? (x>>low) : ((x&((1<<(high+1)) - 1))>>low);
}

__device__ bool fp32_is_zero(uint32_t val)
{
    return (val & (~SIGN_MASK_FP32)) ? 0 : 1;
}

__device__ bool fp32_is_infinity(uint32_t val)
{
    return (val & 0x7FFFFFFF) == 0x7F800000 ? 1 : 0;
}

__device__ bool fp32_is_nan(uint32_t val)
{
    bool isAllExponentBitsSet = ((val & 0x7f800000) == 0x7f800000);
    bool isAnyMantissaBitSet = ((val & 0x007fffff) != 0);
    return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

__device__ int fp_accommodate_rounding( uint32_t intValuePreRounding
                                    , bool roundedMSB, bool roundedLSBs
                                    , unsigned int sign, bool stochastic
                                    , uint32_t lfsrVal, uint32_t discardedAlignedLeft )
{
	uint32_t  result = 0;
	result = intValuePreRounding;

    if (stochastic)
    {  //SR
		if(discardedAlignedLeft >= lfsrVal)
		{
			result = intValuePreRounding + 1;
		}
    }
    else
    { //RNE
		if ((((intValuePreRounding & 0x1) == 1) && (roundedMSB == 1)) ||
			(((intValuePreRounding & 0x1) == 0) && (roundedMSB == 1) && (roundedLSBs == 1)))
		{
			result = intValuePreRounding + 1;
		}
    }
    return result;
}




__device__ int lzcnt(uint32_t bits, uint32_t int_num)
{
    int msb = bits - 1;
    int lsb = 0;
    int i = msb;
    for ( ; i >= lsb; --i) {
        if ((int_num & (1 << i)) != 0) {
            break;
        }
    }
    return bits - i - 1;
}


//=================================
//========== mxfp4 BIT ============
//=================================


//exp_width maximum should be 3 (as fp32)
//man_width minimum should be 1 (as fp32)
//sign is always 1 bit (upper bit)
// Rounding mode: RNE or stochastic

__device__ void make_fp32_to_4bit(float input, uint8_t *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias,  int32_t lfsrVal,bool stochastic)
{
	int inputExponent, inputSign, unbiasedExp = 0;
	uint32_t inputMantissa;
	bool roundedMSB = 0, roundedLSBs = 0;
	int minNormExp = 1 - exp_bias; 
	int maxExp = ((1 << exp_width) - 1) - exp_bias; 
	int minExp = minNormExp - man_width - 1; 
	int32_t exponent_offset = man_width;
	int32_t sign_offset = 3;

	const uint32_t inputUint = *(const uint32_t *)&input;

	inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
	inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
	inputSign = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

	int32_t outputExponent;
	int32_t outputMantissa;
	int32_t outputSign = inputSign;

	int      rc_bit_idx;
    int32_t  shift_val;
    uint32_t discardedAlignedLeft    = 0;

	if (fp32_is_nan(inputUint) || fp32_is_infinity(inputUint))
	{
		// -0 represent inf and nan
		//TODO - merge into the block scale as defined in OCP
		outputExponent = 0x0;
		outputMantissa = 0x0;
		outputSign = 0x1;
	}
	else if (fp32_is_zero(inputUint))
	{
		// return +-0
		outputExponent = 0x0;
		outputMantissa = 0x0;
		outputSign = 0x0;
	}
	else
	{
		// Valid number
		unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
		inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

		if (unbiasedExp > maxExp)
		{
			outputExponent = maxExp + exp_bias;
			if (man_width == 0){
				outputMantissa = 0x0;
			}
			else{
				outputMantissa = sbs(0xff,man_width-1,0);//0x3;
			}
		}
		else if (unbiasedExp < minExp)
		{
			// The result will be either 0 or 0x1
			roundedMSB = 0;
			roundedLSBs = 1;
            if (stochastic) {
                rc_bit_idx = (EXPONENT_OFFSET_FP32 - exponent_offset - 1) + (minNormExp - unbiasedExp);
                shift_val = 31 - rc_bit_idx;
                if (shift_val >= 0)
                    discardedAlignedLeft = inputMantissa << shift_val;
                else if (shift_val >= -24)
                    discardedAlignedLeft = inputMantissa >> (-shift_val);
                else
                    discardedAlignedLeft = 0;
			}
			outputMantissa = fp_accommodate_rounding(0, roundedMSB, roundedLSBs, inputSign, stochastic, lfsrVal, discardedAlignedLeft);
			outputExponent = 0x0;
			if ((man_width == 0) && (outputMantissa == 0x1))
			//go to min normal if no bits to mantisa
			{
				outputMantissa = 0x0;
				outputExponent = 0x1;
			}
			
		}
		else
		{ // minExp <= unbiasedExp <= maxExp
			outputExponent = unbiasedExp;
			rc_bit_idx = (unbiasedExp < minNormExp) ?  ((EXPONENT_OFFSET_FP32 - exponent_offset - 1) + (minNormExp - unbiasedExp)) : (EXPONENT_OFFSET_FP32 - exponent_offset - 1);
			shift_val    = 31 - rc_bit_idx;
			roundedMSB = (((inputMantissa >> rc_bit_idx)) & 0x1) != 0;
			roundedLSBs = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
			discardedAlignedLeft = inputMantissa << shift_val;
			outputMantissa = inputMantissa >> (rc_bit_idx + 1);

			outputMantissa = fp_accommodate_rounding(outputMantissa, 
			roundedMSB, roundedLSBs, inputSign, 
			stochastic, lfsrVal, 
			discardedAlignedLeft);
			
			if (((unbiasedExp < minNormExp) && (outputMantissa & (1 << exponent_offset))) || (outputMantissa & (1 << (exponent_offset + 1))))
			{ // Should handle two cases: 
			  // 1. The number was denormal, and after rounding became normal
			  // 2. The number was rounded to the 1.0 * 2^(next exponent)
				outputExponent = outputExponent + 1;
			}
			if (outputExponent > maxExp)
			{
				// FP4 dont allow nan or inf
				outputExponent = maxExp;
				if (man_width == 0)
				{
				outputMantissa = 0x0;
				}
				else
				{
				outputMantissa = sbs(0xff,man_width-1,0);//0x3;
				}
			}
			else
			{
				if (outputExponent < minNormExp)
				{
					outputExponent = 0x0;
				}
				else
				{
					outputExponent += exp_bias;
				}
				// normalize - leave man_width bits
				if (man_width == 0)
				{
				outputMantissa = 0x0;
				}
				else
				{
				outputMantissa = sbs(outputMantissa, man_width-1, 0);
				}
			}

		}
		if ((outputExponent == 0x0) && (outputMantissa == 0x0))
		{
			outputSign = 0x0;
		}
	}
	*output = 0x0F & (outputMantissa | (outputExponent << exponent_offset) | (outputSign << sign_offset));

}
__device__ bool fp4_is_zero(uint8_t val)
{
	return (val & 0xF) ? 0 : 1;
}

__device__ bool fp4_is_nan_or_inf(uint8_t val)
{

	// -0 represent inf and nan
	return val == 0x8? 1: 0 ;
}

__device__ bool fp4_is_denormal(uint8_t val, uint8_t exponent_offset_fp4)
{ // Do not consider zero as denormal
	bool isAllExponentBitsZero = sbs(val,2,exponent_offset_fp4) == 0;
	bool isAnyMantissaBitSet = (sbs(val,exponent_offset_fp4-1,0) != 0);
	return (isAllExponentBitsZero & isAnyMantissaBitSet);
}

__device__ void make_4bit_to_fp32(uint8_t input, float *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias)
{
	const uint8_t inputUint = input;
	uint32_t *outputUint = (uint32_t *)output;
	int32_t exponent_offset = man_width;
	int32_t sign_offset = 3;

	int32_t inputMantissa = 0x0;

	if (man_width != 0)
	{
		 inputMantissa = sbs(inputUint,man_width-1,0);
	}
	int32_t inputExponent = sbs(inputUint,2,exponent_offset);
	int32_t inputSign = sbs(inputUint,sign_offset,sign_offset);

	int32_t outputExponent;
	int32_t outputMantissa;
	int32_t outputSign = inputSign;

	
	if (fp4_is_zero(inputUint))
	{
		outputExponent = 0x0;
		outputMantissa = 0x0;
	}
	else if (fp4_is_nan_or_inf(inputUint))
	{
		//-0 represnt nan or inf in fp4. return nan
		outputExponent = 0xFF;
		outputMantissa = 0x007FFFFF;
		outputSign = 0;
	}
	else
	{
		outputExponent = inputExponent - exp_bias + EXPONENT_BIAS_FP32;
		int32_t mantissaForAdjustment = inputMantissa;
		if (man_width == 0)
		{
			mantissaForAdjustment = 0x0;
		}
		else if (fp4_is_denormal(inputUint, exponent_offset))
		{
			int shift = lzcnt(exponent_offset, inputMantissa);
			// Shift leading 1 (normalize) and fixup the exponent accordingly
			mantissaForAdjustment = sbs((inputMantissa << (shift + 1)),man_width-1,0);
			outputExponent -= shift;

		}
		// Normal case
		outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP32 - exponent_offset);
	}

	*outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

}


__global__ void Quant4BitKernel(const float* in_data, float* out_data, const int totalElements, const uint8_t exp_width, const uint8_t man_width, const uint8_t exp_bias,  const int32_t* lfsrVal,bool stochastic)
{	
	CUDA_KERNEL_LOOP(i, totalElements){
		uint8_t out_8bit;
		int32_t randn;
		if (stochastic)
			randn = lfsrVal[i];
		else
			randn = 0;
		 make_fp32_to_4bit(in_data[i], &out_8bit, exp_width, man_width, exp_bias,  randn,stochastic);
		 make_4bit_to_fp32(out_8bit, &out_data[i], exp_width, man_width, exp_bias);
	}
}


torch::Tensor mxfp4_cuda(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias,bool stochastic) {
	const auto num_elements = input.numel();

	torch::Tensor output = torch::empty_like(input);
	torch::Tensor rand = torch::empty_like(input, torch::dtype(torch::kInt32));
	if (stochastic){
		rand.random_(STOCH_ROUNDING_MIN, STOCH_ROUNDING_MAX);
	};
	Quant4BitKernel <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements, exp_width, man_width, exp_bias, rand.data_ptr<int32_t>(),stochastic);
	return output;
  }