import torch
import torch.nn as nn

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        if Quantization.is_clip:
            input = torch.clamp(input, 0, 1)
        # quant_v = 255.
        quant_v = Quantization.quant_v
        output = (input * quant_v).round() / quant_v
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self,quant_v = 255.0,is_clip = True):
        super(Quantization, self).__init__()
        Quantization.quant_v = quant_v
        Quantization.is_clip = is_clip

    def forward(self, input):
        return Quant.apply(input)
