import torch
import torch.nn as nn

# class Noise_inner(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input):
#         output = input+torch.nn.init.uniform_(torch.zeros_like(input), -Noise.noise_magnitude, Noise.noise_magnitude)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output



class Noise(nn.Module):
    def __init__(self,noise_magnitude = 1e-4,type = "uniform"):
        super(Noise, self).__init__()
        Noise.noise_magnitude = noise_magnitude
        self.type = type

    def forward(self, input):
        if self.type == 'uniform':
            sign = (torch.bernoulli(torch.ones_like(input)*0.5).cuda(input.device))*2-1
            noise = sign*torch.nn.init.uniform_(torch.zeros_like(input).cuda(input.device), Noise.noise_magnitude/10, Noise.noise_magnitude)
        elif self.type == 'gaussian':
            noise = torch.nn.init.normal_(torch.zeros_like(input).cuda(input.device),0,2)*Noise.noise_magnitude
        
        # noise = sign* Noise.noise_magnitude
        # print(noise)
        # print(noise.max())
        # print(noise.min())
        # print(self.type)
        # exit()

        output = input+noise
        return output


# class RandomQuant(nn.Module):
#     def __init__(self,quant_magnitude = 4e-4):
#         super(Noise, self).__init__()
#         RandomQuant.quant_magnitude = quant_magnitude

#     def forward(self, input):
#         sign = (torch.bernoulli(torch.ones_like(input)*0.5).cuda(input.device))*2-1
#         noise = sign*torch.nn.init.uniform_(torch.zeros_like(input).cuda(input.device), 1e-3, Noise.noise_magnitude)
#         # print(noise)
#         # print(noise.max())
#         # print(noise.min())
#         # exit()
#         output = input+noise
#         return output
