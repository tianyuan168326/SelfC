import torch
import torch.nn as nn
import numpy as np

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            v = (x - target)**2
            # return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            v = torch.sqrt(diff * diff + self.eps)
        else:
            print("reconstruction loss type error!")
            return 0
        return v.mean(-1).mean(-1).mean(-1).mean(-1)

from .spy_flow import *
import random
class MotionFlowLoss(nn.Module):
    def __init__(self):
        super(MotionFlowLoss, self).__init__()
        self.opticFlow_Net = ME_Spynet().cuda()
        # self.opticFlow_Net .requires_grad_(False)

        

    def forward(self, x_lr, target_hr):
        T = 5

        bt,c,h,w = x_lr.size()
        b = bt // T
        x_lr_v = x_lr.reshape(b,T,c,h,w)
        bt,c,h,w = target_hr.size()
        b = bt // T
        target_hr_v = target_hr.reshape(b,T,c,h,w)

        index1 = random.randint(0,3)
        index2 = random.randint(index1,4)
        x_lr_1 = x_lr_v[:,index1]
        x_lr_2 = x_lr_v[:,index2]
        target_hr_1 = target_hr_v[:,index1]
        target_hr_2 = target_hr_v[:,index2]

        target_mv = self.opticFlow_Net(target_hr_2, target_hr_1)
        # print(( (torch_warp(target_hr_1,target_mv) - target_hr_2)**2  ).mean())
        # exit()
        target_mv = torch.nn.functional.upsample(target_mv,scale_factor=(0.25,0.25),mode='area')
        lr_mv = self.opticFlow_Net(x_lr_2, x_lr_1)
        target_mv = torch.cat([target_mv[:, 0:1, :, :] / ((target_mv.size(3) - 1.0) / 2.0), target_mv[:, 1:2, :, :] / ((target_mv.size(2) - 1.0) / 2.0) ], 1)
        lr_mv = torch.cat([lr_mv[:, 0:1, :, :] / ((lr_mv.size(3) - 1.0) / 2.0), lr_mv[:, 1:2, :, :] / ((lr_mv.size(2) - 1.0) / 2.0) ], 1)
        
        # print(target_mv)
        # exit()

        optical_loss = ((target_mv.detach() - lr_mv)**2).mean()
        return optical_loss





# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
