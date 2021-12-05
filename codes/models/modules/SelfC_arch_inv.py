import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
TEMP_LEN = 7

class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


from .Subnet_constructor import DenseBlock,DenseBlock3D,DenseBlockVideoInput,FeatureCalapseBlock,D2DTInput
from torch.autograd.variable import Variable
import torch.distributions as D
class STPNet(nn.Module):
    def __init__(self,opt):
        super(STPNet, self).__init__()
        self.stp_d2d_inner_c = opt["stp_d2d_inner_c"]
        self.stp_temporal_c = opt["stp_temporal_c"]
        self.fh_loss = opt["fh_loss"]
        self.K = opt["gmm_mixture_num"]
        self.stp_blk_num = opt["stp_blk_num"]
        self.condition_func = opt["condition_func"]
        if self.condition_func == "D2DTNet":
            self.blk1 = nn.Sequential(
                D2DTInput(3,12),
                D2DTInput(12,24),
                D2DTInput(24,48),
            )
            self.blk2 = D2DTInput(48,self.stp_temporal_c)
        else:
            self.blk1 = FeatureCalapseBlock(3,12)
            self.blk2 = FeatureCalapseBlock(12,self.stp_temporal_c)
        
        self.hf_dim = 9
        if self.fh_loss == "l2":
            self.tail = [
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(self.stp_temporal_c, self.hf_dim, 1, 1, 0, bias=True)
            ]
            self.tail = nn.Sequential(*self.tail)

        elif self.fh_loss == "gmm":
            MLP_dim = self.stp_temporal_c
            self.tail_gmm = [
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(self.stp_temporal_c, MLP_dim, 1, 1, 0, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(MLP_dim, MLP_dim, 1, 1, 0, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(MLP_dim, self.hf_dim * self.K*3, 1, 1, 0, bias=True)
            ]
            self.tail_gmm = nn.Sequential(*self.tail_gmm)

        
    def forward(self, x):
        b,c,t,h,w = x.size()
        self.b = b
        self.c = c
        self.t = t
        self.h = h
        self.w = w
        b,c,t,h,w = x.size()
        x = x.transpose(1,2)
        x = x.reshape(b*t,c,h,w)
        temp = self.blk1(x)
        temp = self.blk2(temp)
        bt,c,w,h = temp.size()
        t = TEMP_LEN
        b = bt//t
        temp  = temp.reshape(b,t,c,w,h).transpose(1,2)
        if self.fh_loss == "l2":
            temp = self.tail(temp)
            self.parameters = temp
            return 
        if self.fh_loss == "gmm":
            temp = self.tail_gmm(temp)
            self.parameters = temp
            out_param = self.parameters

            b,c,t,h,w = out_param.size()
            out_param = out_param.reshape(b,self.hf_dim,self.K,3,t,h,w)
            pi = F.softmax(out_param[:,:,:,0],dim=1)
            log_scale = torch.clamp(out_param[:,:,:,1],-7,7)
            mean = out_param[:,:,:,2]
            v = torch.zeros_like(out_param[:,:,0,0]).cuda(out_param.device)
            for i in range(self.K):
                v+=pi[:,:,i]* self.reparametrize(mean[:,:,i], log_scale[:,:,i]) # 重新参数化成正态分布
            self.gmm_v = v
            out_param = self.parameters
            b,c,t,h,w = out_param.size()
            out_param = out_param.reshape(b,self.hf_dim,self.K,3,t,h,w)
            out_param = out_param.permute(0,1,4,5,6, 2,3).reshape(-1,self.K,3)
            pi = F.softmax(out_param[:,:,0],dim=1)
            log_scale = torch.clamp(out_param[:,:,2],-7,7)
            gm_p =  torch.stack((pi,out_param[:,:,1],log_scale),dim=-1)
            weight = gm_p[:,:,0]
            mean = gm_p[:,:,1]
            log_var = gm_p[:,:,2]
            mix = D.Categorical(weight)
            comp = D.Normal(mean,torch.exp(log_var))
            self.gmm = D.MixtureSameFamily(mix, comp)
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def neg_llh(self,hf):
        b,c,t,h,w = hf.size()
        if self.fh_loss == "gmm":
            hf = hf.reshape(-1)
            return -self.gmm.log_prob(hf)
        elif self.fh_loss == "l2":
            # return torch.sum((hf-self.parameters)**2)/(b*t)
            return torch.mean((hf-self.parameters)**2)
    def sample(self):
        if self.fh_loss == "gmm":
            return self.gmm_v
        elif self.fh_loss == "l2":
            return self.parameters

class SelfCInvNetUnshared(nn.Module):
    def __init__(self, opt, channel_in, channel_out,subnet_type, block_num, down_num):
        super(SelfCInvNetUnshared, self).__init__()
        operations = []
        current_channel = channel_in
        subnet_constructor = subnet(subnet_type,"xavier")
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        operations1 = []
        block_num = [2,2]
        current_channel = channel_in
        subnet_constructor = subnet(subnet_type,"xavier")
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations1.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations1.append(b)

        self.operations1 = nn.ModuleList(operations1)

        self.stp_net = STPNet(opt)

    def forward(self, x, rev=False, cal_jacobian=False,lr_before_distor = None):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            bt,c,h,w = out.size()
            t = 7
            b = bt//t
            out = out.reshape(b,t,c,h,w).transpose(1,2)
            lf = out[:,0:3]
            hf = out[:,3:]
            out = out.transpose(1,2).reshape(b*t,c,h,w)
            self.stp_net(lf)
            loss_c = self.stp_net.neg_llh(hf)
            return out,loss_c
        else:
            bt,c,h,w = out.size()
            t = 7
            b = bt//t
            out = out.reshape(b,t,c,h,w).transpose(1,2)
            lr_input = x[:,0:3]

            bt,c,h,w = lr_input.size()
            t = 7
            b = bt//t
            lr_input = lr_input.reshape(b,t,c,h,w).transpose(1,2)

            self.stp_net(lr_input)
            recon_hf = self.stp_net.sample()
            out = torch.cat((lr_input,recon_hf),dim=1)
            out = out.transpose(1,2).reshape(b*t,out.size(1),h,w)
            for op in reversed(self.operations1):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            return out,recon_hf.transpose(1,2).reshape(b*t,-1,h,w)
        if cal_jacobian:
            return out, jacobian
        else:
            return out
class SelfCInvNet(nn.Module):
    def __init__(self, opt, channel_in, channel_out,subnet_type, block_num, down_num):
        super(SelfCInvNet, self).__init__()
        operations = []
        current_channel = channel_in
        subnet_constructor = subnet(subnet_type,"xavier")
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)



        self.stp_net = STPNet(opt)

    def forward(self, x, rev=False, cal_jacobian=False,lr_before_distor = None):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            bt,c,h,w = out.size()
            t = TEMP_LEN
            b = bt//t
            out = out.reshape(b,t,c,h,w).transpose(1,2)
            lf = out[:,0:3]
            hf = out[:,3:]
            out = out.transpose(1,2).reshape(b*t,c,h,w)
            self.stp_net(lf)
            loss_c = self.stp_net.neg_llh(hf)
            return out,loss_c
        else:
            bt,c,h,w = out.size()
            t = TEMP_LEN
            b = bt//t
            out = out.reshape(b,t,c,h,w).transpose(1,2)
            lr_input = x[:,0:3]

            bt,c,h,w = lr_input.size()
            b = bt//t
            lr_input = lr_input.reshape(b,t,c,h,w).transpose(1,2)

            self.stp_net(lr_input)
            recon_hf = self.stp_net.sample()
            out = torch.cat((lr_input,recon_hf),dim=1)
            out = out.transpose(1,2).reshape(b*t,out.size(1),h,w)
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            return out,recon_hf.transpose(1,2).reshape(b*t,-1,h,w)
        if cal_jacobian:
            return out, jacobian
        else:
            return out
from models.modules.Subnet_constructor import subnet

import models.modules.module_util as mutil
