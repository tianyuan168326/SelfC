import sys
# sys.path.append("/media/ps/SSD/tianyuan/SelfC")
# sys.path.append("/data_video/code/SelfC")
# from ImageCompression.model import *
# from ImageCompression.models.basics import *
# from PyTorchVideoCompression.DVC.net import *
# from PyTorchVideoCompression.DVC.subnet.basics import *
# from net import *
# from subnet.basics import *
import utils.util as util
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.Quantization import Quantization
from models.modules.Noise import Noise
from models.modules.Quantization_BPG import Quantization_BPG
from models.modules.Quantization_video_compression import Quantization_H265
from models.modules.Quantization_h265_rgb_stream import Quantization_H265_Stream
# from models.modules.Quantization_h265_suggrogate import Quantization_H265_Suggrogate
# from models.modules.Quantization_h265_suggrogate_correct import Quantization_H265_Suggrogate
from models.modules.Quantization_h265_suggrogate_correlation1 import Quantization_H265_Suggrogate
from global_var import *
import torchvision.ops
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


import torch.nn as nn

class PixelUnshuffle(nn.Module):
	def __init__(self, scale=4):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		N, C, H, W = x.size()
		S = self.scale
		# (N, C, H//bs, bs, W//bs, bs)
		x = x.view(N, C, H // S, S, W // S, S)  
		# (N, bs, bs, C, H//bs, W//bs)
		x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  
		# (N, C*bs^2, H//bs, W//bs)
		x = x.view(N, C * S * S, H // S, W // S)  
		return x
TEMP_LEN = 5
class FrequencyAnalyzer(nn.Module):
	def __init__(self, channel_in,k=2):
		super(FrequencyAnalyzer, self).__init__()
		# k = 4
		self.bicubic_down = nn.Upsample(scale_factor=(1/k,1/k),mode = "area")
		self.pixel_unshuffle  = PixelUnshuffle(k)

		self.bicubic_up = nn.Upsample(scale_factor=(k,k),mode = "area")
		self.pixel_shuffle  = nn.PixelShuffle(k)


	def forward(self, x, rev=False):
		#### forward
		if not rev:
			component_low_f = self.bicubic_down(x)
			component_high_f = self.pixel_unshuffle(x - self.bicubic_up(component_low_f))
			return torch.cat((component_low_f,component_high_f),dim = 1)
		else:
			component_low_f = x[:,0:3]
			component_high_f = x[:,3:]
			return self.bicubic_up(component_low_f) + self.pixel_shuffle(component_high_f)

from .Subnet_constructor import DenseBlock,DenseBlock3D,DenseBlockVideoInput,FeatureCalapseBlock,D2DTInput
from torch.autograd.variable import Variable
import torch.distributions as D
class GlobalAgg(nn.Module):
	def __init__(self,c):
		super(GlobalAgg, self).__init__()
		self.fc = nn.Linear(32*32,1)
		self.proj1 = nn.Conv2d(c,c,1,1,0)
		self.proj2 = nn.Linear(c,c)
		self.proj3 = nn.Linear(c,c)

	def forward(self, x):
		### 64 w h 
		x_proj1 = self.proj1(x)

		B,C,H,W = x.size()
		x_down_sample = F.adaptive_avg_pool2d(x,output_size = (32,32))
		x_down_sample = x_down_sample.reshape(B,C,32*32)
		x_down_sample = self.fc(x_down_sample).squeeze()
		x_down_sample_video = x_down_sample.reshape(B//TEMP_LEN,TEMP_LEN,C)
		x_down_sample_video_proj2 = self.proj2(x_down_sample_video)
		x_down_sample_video_proj3 = self.proj3(x_down_sample_video)
		temporal_weight_matrix = torch.matmul(x_down_sample_video_proj2,x_down_sample_video_proj3.transpose(1,2))
		#### T * T
		temporal_weight_matrix = F.softmax(temporal_weight_matrix/C, dim=-1)

		x_proj1 = x_proj1.reshape(B//TEMP_LEN,TEMP_LEN,C,H,W)
		x_proj1 = x_proj1.permute(0,2,3,4,1).reshape(B//TEMP_LEN,C*H*W,TEMP_LEN)
		weighted_feature = torch.matmul(x_proj1,temporal_weight_matrix) ## b (chw) t

		return x + weighted_feature.reshape(B//TEMP_LEN,C,H,W,TEMP_LEN).\
			permute(0,4,1,2,3).reshape(B,C,H,W)


class GroupedGlobalDeformAgg(nn.Module):
	def __init__(self,c,T = 3):
		super(GroupedGlobalDeformAgg, self).__init__()
		self.g = 4
		self.global_context_per_group = T*(c//self.g)
		self.global_context_reallocator = nn.Sequential(
		nn.Conv2d(
			self.global_context_per_group,
			self.global_context_per_group,
			3,1,1,bias=True
		),
		nn.LeakyReLU(0.2),
		nn.Conv2d(
			self.global_context_per_group,
			self.global_context_per_group,
			3,1,1,bias=True
		)
		)
		nn.init.constant_(self.global_context_reallocator[2].weight, 0.)
		nn.init.constant_(self.global_context_reallocator[2].bias, 0.)
		in_channels = c
		kernel_size=3
		stride = 1
		self.padding = 1
		self.offset_conv = nn.Conv2d(in_channels, 
									 2 * kernel_size * kernel_size * T,
									 kernel_size=kernel_size, 
									 stride=stride,
									 padding=self.padding, 
									 bias=True)

		nn.init.constant_(self.offset_conv.weight, 0.)
		nn.init.constant_(self.offset_conv.bias, 0.)
		
		self.modulator_conv = nn.Conv2d(in_channels, 
									 1 * kernel_size * kernel_size* T,
									 kernel_size=kernel_size, 
									 stride=stride,
									 padding=self.padding, 
									 bias=True)

		nn.init.constant_(self.modulator_conv.weight, 0.)
		nn.init.constant_(self.modulator_conv.bias, 0.)
		
		self.regular_conv = nn.Conv2d(in_channels=c,
									  out_channels=c,
									  kernel_size=kernel_size,
									  stride=stride,
									  padding=self.padding,
									  bias=True)
		nn.init.constant_(self.regular_conv.weight, 0.)
		nn.init.constant_(self.regular_conv.bias, 0.)
		# self.proj1 = nn.Conv2d(in_channels=c,
		#                               out_channels=c,
		#                               kernel_size=1,
		#                               stride=1,
		#                               padding=0,
		#                               bias=False)
		# nn.init.constant_(self.proj1.weight, 0.)
		
	def forward(self, x):
		### BT 64 w h 
		TEMP_LEN = GlobalVar.get_Temporal_LEN()
		bt,c,h,w = x.size()
		t = TEMP_LEN
		b = bt//t
		g = self.g
		x_grouped = x.reshape(b,t,g,c//g,h,w)
		x_rearranged = x.reshape(b,t,g,c//g,h,w).transpose(1,2)\
			.reshape(b*g,self.global_context_per_group,h,w)
		x_global_enhanced = self.global_context_reallocator(x_rearranged)
		x_enhanced = x_rearranged + x_global_enhanced ### global feature add to the original

		x_enhanced = x_enhanced.reshape(b,g,t,c//g,h,w).permute(0,2,1,3,4,5)\
			.reshape(b*t,c,h,w)
		

		x_stacked = x_enhanced
		offset = self.offset_conv(x_stacked)#.clamp(-max_offset, max_offset) ## b T (T 2KK ) h w
		modulator = 2. * torch.sigmoid(self.modulator_conv(x_stacked))## b T (T KK ) h w
		offset = offset.reshape(b*(TEMP_LEN*TEMP_LEN), -1,h,w)
		modulator = modulator.reshape(b*(TEMP_LEN*TEMP_LEN), -1,h,w)
		x_repeat = x_enhanced.unsqueeze(1).repeat(1,TEMP_LEN,1,1,1).reshape(b*(TEMP_LEN*TEMP_LEN),c,h,w)
		x_repeat = torchvision.ops.deform_conv2d(input=x_repeat, 
										  offset=offset, 
										  weight=self.regular_conv.weight, 
										  bias=self.regular_conv.bias, 
										  padding=self.padding,
										  mask=modulator,
										  stride=1,
										  )
		x_repeat = x_repeat.reshape(b*(TEMP_LEN),TEMP_LEN,c,h,w)
		x_repeat = x_repeat.sum(1)
		# x_weight = (self.proj1(x_enhanced) )  
		

		return x_enhanced +  x_repeat


		

class STPNet(nn.Module):
	def __init__(self,opt):
		super(STPNet, self).__init__()
		self.global_module = opt["global_module"]
		self.stp_blk_num = opt["stp_blk_num"]
		self.fh_loss = opt["fh_loss"]
		self.scale = opt["scale"]
		self.K = opt["gmm_k"]
		self.stp_blk_num = self.stp_blk_num-2
		c = 32
		c = opt["stp_hidden_c"]
		self.local_m1 = D2DTInput(3,c,INN_init=False)
		self.local_m2 = D2DTInput(c,c,INN_init=False)
		
		if self.global_module == 'nonlocal':
			self.global_m1 = GlobalAgg(c)
			self.global_m2 = GlobalAgg(c)
		if self.global_module == 'deform':
			self.global_m1 = DeformConvAgg(c)
			self.global_m2 = DeformConvAgg(c)
		if self.global_module == 'grouped_global_deform':
			self.global_m1 = GroupedGlobalDeformAgg(c)
			self.global_m2 = GroupedGlobalDeformAgg(c)
		self.other_stp_modules = []
		for i in range(self.stp_blk_num):
			self.other_stp_modules +=[D2DTInput(c,c,INN_init=False)]
			if self.global_module == 'nonlocal':
				self.other_stp_modules +=[GlobalAgg(c)]
			if self.global_module == 'deform':
				self.other_stp_modules +=[DeformConvAgg(c)]
			if self.global_module == 'grouped_global_deform':
				self.other_stp_modules +=[GroupedGlobalDeformAgg(c)] 
			
		self.other_stp_modules = nn.Sequential(*self.other_stp_modules)
		
		self.hf_dim = 3*(self.scale**2)

		if self.fh_loss == "l2":
			self.tail_gmm = [
				nn.LeakyReLU(negative_slope=0.2, inplace=True),
				nn.Conv3d(c, self.hf_dim, 1, 1, 0, bias=True)
			]
			self.tail_gmm = nn.Sequential(*self.tail_gmm)
		elif self.fh_loss == "gmm":
			MLP_dim = c
			self.tail_gmm = [
				nn.LeakyReLU(negative_slope=0.2, inplace=True),
				nn.Conv3d(c, MLP_dim*2, 1, 1, 0, bias=True),
				nn.LeakyReLU(negative_slope=0.2, inplace=True),
				nn.Conv3d(MLP_dim*2, MLP_dim*4, 1, 1, 0, bias=True),
				nn.LeakyReLU(negative_slope=0.2, inplace=True),
				nn.Conv3d(MLP_dim*4, self.hf_dim * self.K*3, 1, 1, 0, bias=True)
			]
			self.tail_gmm = nn.Sequential(*self.tail_gmm)
		elif self.fh_loss == "gmm_thin":
			MLP_dim = c
			self.tail_gmm = [
				nn.LeakyReLU(negative_slope=0.2, inplace=True),
				nn.Conv3d(c, MLP_dim, 1, 1, 0, bias=True),
				nn.ReLU( inplace=True),
				nn.Conv3d(MLP_dim, MLP_dim, 1, 1, 0, bias=True),
				nn.ReLU( inplace=True),
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
		temp = self.local_m1(x)
		if self.global_module:
			temp = self.global_m1(temp)
		temp = self.local_m2(temp)
		if self.global_module:
			temp = self.global_m2(temp)
		temp = self.other_stp_modules(temp)
		bt,c,w,h = temp.size()
		t = GlobalVar.get_Temporal_LEN()
		b = bt//t
		temp  = temp.reshape(b,t,c,w,h).transpose(1,2)
		self.parameters = self.tail_gmm(temp)
		if self.fh_loss == "l2":
			return

		out_param = self.parameters

		b,c,t,h,w = out_param.size()
		out_param = out_param.reshape(b,self.hf_dim,self.K,3,t,h,w)
		pi = F.softmax(out_param[:,:,:,0],dim=1)
		log_scale = torch.clamp(out_param[:,:,:,1],-7,7)
		mean = out_param[:,:,:,2]
		
		v=pi[:,:,:]* self.reparametrize(mean[:,:,:], log_scale[:,:,:]) # 重新参数化成正态分布
		v = v.sum(2)

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
		# print(weight)
		# print(mean)
		# print(log_var)
		mix = D.Categorical(weight)
		comp = D.Normal(mean,torch.exp(log_var))
		self.gmm = D.MixtureSameFamily(mix, comp)
	def reparametrize(self, mu, logvar):
		std = torch.exp(logvar)
		eps = torch.cuda.FloatTensor(std.size()).fill_(0.0)
		eps.normal_()
		x=eps.mul(std).add_(mu)
		return x
	def neg_llh(self,hf):
		b,c,t,h,w = hf.size()
		if self.fh_loss in ["gmm",'gmm_thin']:
			hf = hf.reshape(-1)
			return -self.gmm.log_prob(hf)
		elif self.fh_loss == "l2":
			# return torch.sum((hf-self.parameters)**2)/(b*t)
			return torch.mean((hf-self.parameters)**2)
	def sample(self):
		if self.fh_loss in ["gmm",'gmm_thin']:
			return self.gmm_v
		elif self.fh_loss == "l2":
			return self.parameters

from utils.util import seg_add_pad,seg_remove_pad
from .Subnet_constructor import *
import time
class SelfCInvNet(nn.Module):
	def __init__(self, opt, channel_in, channel_out,subnet_type, block_num, down_num,all_opt = None):
		super(SelfCInvNet, self).__init__()
		operations = []
		current_channel = channel_in
		subnet_constructor = subnet(subnet_type,"xavier")
		b = FrequencyAnalyzer(current_channel,2)
		operations.append(b)
		current_channel *= (2**2+1)
		# print("down_num",down_num)
		for i in range(down_num):
			for j in range(block_num[i]):
				b = InvBlockExp(subnet_constructor, current_channel, channel_out)
				operations.append(b)
		self.operations = nn.ModuleList(operations)
		self.stp_net = STPNet(opt)
		if opt["deart_net"]:
			deart_hidden_c = 32
			self.deart_net  = nn.Sequential(
				D2DTInput(3,deart_hidden_c,INN_init=False,is_res=True),
				GroupedGlobalDeformAgg(deart_hidden_c),
				D2DTInput(deart_hidden_c,3,INN_init=False,is_res=True)
			)

		self.all_opt = all_opt
		self.opt = opt

		self.net_opt = all_opt["network_G"]
		self.scale = all_opt["scale"]
		self.Quantization_H265_Stream = Quantization_H265_Stream(self.opt["h265_q"],self.opt["h265_keyint"],all_opt["scale"],self.opt)
		if self.all_opt["train"]:
			train_len = all_opt["datasets"]["train"]["video_len"]
			if self.all_opt["train"]["h265_sug"]:
				self.Quantization_H265_Suggrogate = Quantization_H265_Suggrogate(self.net_opt['lambda_corr'],self.net_opt["h265_q"],train_len,all_opt["scale"])
			else:
				self.Quantization_H265 = Quantization_H265(self.opt["h265_q"],train_len,all_opt["scale"])
		self.Quantization = Quantization()
	   

	def noise_video_suggrate(self,LR):
		# LR = self.output[:, :3, :, :]
		# bt,c,h,w = LR.size()
		# b,t  = bt//7,7
		# LR = LR.reshape(b,t,c,h,w)
		b,c,t,h,w = LR.size()
		LR = LR.transpose(1,2)
		LR_flatten = LR.reshape(b*t,c,h,w)
		zero = torch.Tensor([0]).cuda(LR.device)
		mimick_loss = zero
		
		if GlobalVar.get_Istrain():
			LR_flatten_quantized = self.Quantization(LR_flatten)
			if self.all_opt["train"]["noise_type"] == 'h265':
				if self.all_opt["train"]["h265_sug"]:
					LR_FUCK,mimick_loss  = self.Quantization_H265_Suggrogate(LR_flatten_quantized)
				else:
					LR_FUCK,mimick_loss  = self.Quantization_H265(LR_flatten_quantized)
			else:
				LR_FUCK  = self.Noiser(LR_flatten_quantized)
			
			# img_distor_loss = torch.mean((LR_flatten_quantized.clone().detach()-LR_flatten)**2)
			# img_distor_loss = img_distor_loss*256*256*3
			# if mimick_loss:
			#     img_distor_loss = img_distor_loss+ mimick_loss
			img_bpp = zero
		return LR_flatten,LR_FUCK,zero,zero,mimick_loss,img_bpp

	def forward(self, x, rev=False, cal_jacobian=False,lr_before_distor = None):
		if GlobalVar.get_Istrain():
			return self.forward_train(x,rev,cal_jacobian,lr_before_distor)
		else:
			torch.cuda.empty_cache()
			return self.forward_test(x,rev,cal_jacobian,lr_before_distor)
	def forward_train(self, x, rev=False, cal_jacobian=False,lr_before_distor = None):
		
		out = x
		jacobian = 0

		if not rev:
			# if False:
			for op in self.operations:
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)

			bt,c,h,w = out.size()
			t = GlobalVar.get_Temporal_LEN()
			b = bt//t
			out = out.reshape(b,t,c,h,w).transpose(1,2)
			lf = out[:,0:3]
			hf = out[:,3:]

			LR,LR_FUCK,distortion_loss,distribution_loss,img_distor,img_distri = self.noise_video_suggrate(lf)
			loss_c = lf.mean() *0
			if LR_FUCK.size(0) == 0:
				print("compressed video length zero,passng....")
				LR = F.upsample(x,scale_factor=(0.5,0.5))
				LR_FUCK = LR
			return LR,LR_FUCK,loss_c,distortion_loss,distribution_loss,img_distor,img_distri
		else:
			bt,c,h,w = out.size()
			t = GlobalVar.get_Temporal_LEN()
			b = bt//t
			out = out.reshape(b,t,c,h,w).transpose(1,2)
			lr_input = x[:,0:3]
			if self.opt["deart_net"]:
				lr_input = self.deart_net(lr_input)
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
			# out_final = self.tail_net(out)+out
			return out

	def forward_test(self, x, rev=False, cal_jacobian=False,lr_before_distor = None):
		
		out = x.cpu()
		jacobian = 0
		#### if training clip length changed,please reset this value:Seg_Len
		Seg_Len = 3
		divide_width_num = 2
		divide_height_num = 2
		if not rev:
			bt,c,h,w = out.size()
			t = GlobalVar.get_Temporal_LEN()
			b = bt//t
			out_video = out.reshape(b,t,c,h,w)
			
			# print(b,t,c,h,w)
			# if False:
			out_video,pad_num = seg_add_pad(out_video,Seg_Len)
			b,seg_num,seg_len,c,h,w = out_video.size()
			# print(b,seg_num,seg_len,c,h,w)
			# exit()

			outs = []
			old_t_len = GlobalVar.get_Temporal_LEN()
			GlobalVar.set_Temporal_LEN(Seg_Len)
			self.Quantization_H265_Stream.open_writer(out.device,w//self.all_opt["scale"],h//self.all_opt["scale"])
			for seg_i in range(seg_num):
				c,h,w  = out_video.size(-3),out_video.size(-2),out_video.size(-1)
				out = out_video[:,seg_i].reshape(-1,c,h,w)

				## devide frame by half width
				out = out.cuda(0)
				print(out.size(),GlobalVar.get_Temporal_LEN())
				T1 = time.time()
				outs = []

				for i in range(divide_width_num):
					out_temp = out[:,:,:,i*(w//divide_width_num):(i+1)*(w//divide_width_num)]
					for op in self.operations:
						out_temp = op.forward(out_temp, rev)
					outs += [out_temp]
				out = torch.cat(outs,dim=-1)

				T2 = time.time()
				print('encode time {} ms for {} frames'.format ((T2 - T1)*1000,Seg_Len))
				out = out.detach().to("cpu")
				##only LR chnnels
				out = out[:,0:3]
				# print("out",out.size())
				self.Quantization_H265_Stream.write_multi_frames(out)
				# c,h,w = out.size(-3),out.size(-2),out.size(-1)
				# out = out.reshape(b,seg_len,c,h,w)
				# outs += [out]
			GlobalVar.set_Temporal_LEN(old_t_len)
			img_distri = self.Quantization_H265_Stream.close_writer()
			self.Quantization_H265_Stream.open_reader()
			outs = []
			for seg_i in range(seg_num):
				v_seg = self.Quantization_H265_Stream.read_multi_frames(Seg_Len)
				outs+=[v_seg]
			out = torch.cat(outs,dim=0)
			h,w = out.size(-2),out.size(-1)
			out = out.reshape(b,-1,Seg_Len,3,h,w)
			out = seg_remove_pad(out,pad_num,Seg_Len)

			LR = LR_FUCK = out.reshape(-1,3,h,w)
			zero = torch.Tensor([0])
			return LR,LR_FUCK,zero,zero,zero,zero,img_distri
		else:
			bt,c,h,w = out.size()
			t = GlobalVar.get_Temporal_LEN()
			b = bt//t
			out_video = out.reshape(b,t,c,h,w)
			# if False:
			out_video,pad_num = seg_add_pad(out_video,Seg_Len)
			b,seg_num,seg_len,c,h,w = out_video.size()
			outs = []
			old_t_len = GlobalVar.get_Temporal_LEN()
			GlobalVar.set_Temporal_LEN(Seg_Len)
			for seg_i in range(seg_num):
				c,h,w  = out_video.size(-3),out_video.size(-2),out_video.size(-1)
				out = out_video[:,seg_i].reshape(-1,c,h,w)
				out = out.cuda(0)
				bt,c,h,w = out.size()
				t = GlobalVar.get_Temporal_LEN()
				b = bt//t
				out = out.reshape(b,t,c,h,w).transpose(1,2)
				lr_input = out[:,0:3] ### b c t h w
				# util.save_img(util.tensor2img(lr_input[:,:,0]), "f1.jpg")
				# exit()
				# lr_input = lr_input.squeeze()##eliminate the t dim
				T11 = time.time()
				outs_here = []
				b,c,t,h,w = lr_input.size()
				hd = h//divide_height_num
				wd = w//divide_width_num
				lr_input = lr_input.reshape(b,c,t,divide_height_num,hd,divide_width_num,wd).\
				permute(0,3,5,1,2,4,6).\
				reshape(b,divide_height_num*divide_width_num,c,t,hd,wd)
				util.save_img(util.tensor2img(lr_input[:,0,:,0]), "out_temp.jpg")
				# exit()

				for i in range(lr_input.size(1)):
					lr_input_temp = lr_input[:,i]
					# lr_input[:,:,:,:,i*(w//divide_width_num):(i+1)*(w//divide_width_num)]
					print(lr_input_temp.size())
					self.stp_net(lr_input_temp)
					recon_hf_temp = self.stp_net.sample()
					
					out_temp = torch.cat((lr_input_temp,recon_hf_temp),dim=1)
					out_temp = out_temp.transpose(1,2).reshape(b*t,out_temp.size(1),hd,wd)
					
					for op in reversed(self.operations):
						out_temp = op.forward(out_temp, rev)
					
					outs_here +=[out_temp]
				out = torch.stack(outs_here, dim=1)### bt p c h w
				print("fuck",out.size())
				
				hd,wd = out.size(-2),out.size(-1)
				out = out.reshape(b,t,divide_height_num,divide_width_num,c,hd,wd).\
				permute(0,4,1,2 ,5, 3,  6).\
				reshape(b,c,t,hd*divide_height_num,wd*divide_width_num).\
				transpose(1,2)### b t c h w
				

				T21 = time.time()
				print('decode time {} ms for {} frames'.format ((T21 - T11)*1000,Seg_Len))
				
				out = out.detach().to("cpu")
				# print("out",out.size())
				# out_final = self.tail_net(out)+out
				# c,h,w = out.size(-3),out.size(-2),out.size(-1)
				# out = out.reshape(b,seg_len,c,h,w)
				outs += [out]
			GlobalVar.set_Temporal_LEN(old_t_len)
			out = torch.stack(outs,dim=1)
			out = seg_remove_pad(out,pad_num,Seg_Len)
			b,t,c,h,w = out.size()
			return out.reshape(-1,c,h,w)

from models.modules.Subnet_constructor import subnet

import models.modules.module_util as mutil
