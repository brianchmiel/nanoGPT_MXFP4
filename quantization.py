import torch
import torch.nn.functional as F

from mxfp4_kernel.mxfp4_operation import mxfp4_quantization_func
def compute_scaling_factor(amax, fp_max):

    scale = torch.ones_like(amax)
    exp = torch.floor(torch.log2(fp_max / amax)) 
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(exp), sf, scale)
    sf = torch.where(exp < 0, 1 / sf, sf)
    return sf


def mx_quantize(x,exp_bits,stochastic = False):
    output = x.float().clone()
    exp_bias = 1
    man_width = 3 - exp_bits
    mxFp = (2 - (2 ** (-1 * man_width))) * (2 ** ((2**exp_bits)-1-exp_bias)) #No Inf max
    scale = compute_scaling_factor(output.abs().max(dim=-1, keepdim=True)[0],mxFp)
    output = mxfp4_quantization_func((output*scale),exp=exp_bits,man=man_width,bias=exp_bias,stochastic=stochastic,clip=True)/scale
    return output



class MXFp4QuantLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MXFp4QuantLinear, self).__init__(in_features, out_features, bias)
        self.blocksize  =32

    def forward(self, input):

        return MXFp4QuantLinearFunc.apply(input,self.weight,self.bias)



class MXFp4QuantLinearFunc(torch.autograd.Function):
    """Linear semi-top level module
    """
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        blocksize=32
        #Input quantization 121 RDN
        qinput = mx_quantize(input.reshape(-1,blocksize),2,False)
        qinput = qinput.reshape(input.shape)
        #Weight quantization 121 RDN
        qweight = mx_quantize(weight.reshape(-1,blocksize),2,False)
        qweight = qweight.reshape(weight.shape)

        output = F.linear(qinput, qweight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        blocksize = 32
        grad_output = grad_output.contiguous()

        #### GRAD INPUT ####
        #Weight quantization 121 RDN according to first axis
        qos_weight = torch.movedim(weight,-1,-2)
        sh = qos_weight.shape
        qos_weight = mx_quantize(qos_weight.reshape(-1,blocksize),2,False)
        qos_weight = torch.movedim(qos_weight.reshape(sh),-1,-2)

        #Grad output quantization 130 SR 
        qos_grad_output = mx_quantize(grad_output.reshape(-1,blocksize),3,True)
        qos_grad_output = qos_grad_output.reshape(grad_output.shape)

        grad_input = qos_grad_output.matmul(qos_weight.contiguous())


        #### GRAD WEIGHT ####

        #input quantization 121 RDN according to first axis
        qex_input = torch.movedim(input,-1,-2)
        sh = qex_input.shape
        qex_input =  mx_quantize(qex_input.reshape(-1,blocksize),2,False)
        qex_input = torch.movedim(qex_input.reshape(sh),-1,-2)
        qex_input = qex_input.contiguous().reshape(-1,weight.shape[1])

        #Grad output quantization 130 SR 
        qex_grad_output = torch.movedim(grad_output,-1,-2)
        sh = qex_grad_output.shape
        qex_grad_output = mx_quantize(qex_grad_output.reshape(-1,blocksize),3,True)
        qex_grad_output = torch.movedim(qex_grad_output.reshape(sh),-1,-2)

        qex_grad_output = qex_grad_output.reshape(-1,weight.shape[0])

        grad_weight = qex_grad_output.t().matmul(qex_input)

        #### GRAD BIAS ####

        grad_bias = grad_output.sum(dim=0) if use_bias else None


        return grad_input, grad_weight, grad_bias