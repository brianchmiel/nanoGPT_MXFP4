#include <torch/extension.h>
//declaration


torch::Tensor mxfp4_cuda(torch::Tensor input, 
const int exp_width, 
const int man_width, 
const int exp_bias,
bool stochastic);


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

torch::Tensor mxfp4_quantizer(
    torch::Tensor input, 
const int exp_width, 
const int man_width, 
const int exp_bias,
bool stochastic)
{
  CHECK_INPUT(input);

  return mxfp4_cuda(input, exp_width, man_width,exp_bias,stochastic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("mxfp4_quantizer", &mxfp4_quantizer, "mxfp4 (CUDA)");

}