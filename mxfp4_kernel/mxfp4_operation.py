
import torch
from torch.autograd import Function

from mxfp4Quant._C import mxfp4_quantizer


def mxfp4_quantization_func(input,exp,man,bias=None,stochastic=False,clip=True):
    assert exp + man == 3, "sum of exponent and mantissa bits should be 3"
    assert exp <= 3 , "maximum exponent bits = 3"
    assert exp > 0 , "minimum exponent bits = 1"
    assert man >= 0 , "minimum mantissa bits = 0"

    #Todo - extend kernel for half and bfloat
    is_half = False
    is_bfloat = False
    if input.dtype == torch.half:
        input = input.float()
        is_half = True
    
    if input.dtype == torch.bfloat16:
        input = input.float()
        is_bfloat = True

    #default exponent bias
    if bias is None:
        bias = 2 ** (exp-1) - 1

    if clip:
        mxFp = (2 - (2 ** (-1 * man))) * (2 ** ((2**exp)-1-bias)) #No Inf max
        input = torch.clamp(input,(-1)*mxFp,mxFp)


    out = mxfp4_quantizer(input,exp,man,bias,stochastic)

    if is_half:
        out = out.half()
    if is_bfloat:
        out = out.bfloat16()
    return out


