import sys
sys.path.append("/data_video/code/SelfC/PyTorchVideoCompression/DVC")
# sys.path.append("/media/ps/SSD/tianyuan/SelfC/PyTorchVideoCompression/DVC")
# from net import *
# from subnet.basics import *
import logging
from collections import OrderedDict
from .Guassian import Guassian_downsample

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
import utils.util as util

logger = logging.getLogger('base')

class SelfCModel(BaseModel):
	def __init__(self, opt):
		super(SelfCModel, self).__init__(opt)
		self.opt = opt

		if opt['dist']:
			self.rank = torch.distributed.get_rank()
		else:
			self.rank = -1  # non dist training
		train_opt = opt['train']
		test_opt = opt['test']
		self.train_opt = train_opt
		self.test_opt = test_opt
		# self.netG = networks.define_G(opt).to(self.device)
		# self.netG = networks.define_G(opt).to(self.device)
		# if opt['dist']:
		#     self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
		# else:
		#     self.netG = self.netG.to(self.device)
		#     # if len(self.device)>1:
		#     if opt['multi_gpu']:
		#         self.netG = DataParallel(self.netG)
		#     pass
		self.netG = networks.define_G(opt).to(self.device)
		if opt['dist']:
			self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
		else:
			self.netG = DataParallel(self.netG)
		# print network
		self.print_network()
		self.load()

		# self.Quantization = Quantization()
		self.Reconstruction_back = ReconstructionLoss(losstype="l1")
		if self.is_train:
			self.netG.train()

			# loss
			self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
			self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])


			# optimizers
			wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
			optim_params = []
			for k, v in self.netG.named_parameters():
				# if not "deart_net" in k:
				# #     optim_params.append({"params":v, "multiplier":0.01})
				#     continue
				if v.requires_grad:
					optim_params.append({"params":v, "multiplier":1.0})
					# optim_params.append(v)
				else:
					if self.rank <= 0:
						logger.warning('Params [{:s}] will not optimize.'.format(k))
			self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
												weight_decay=wd_G,
												betas=(train_opt['beta1'], train_opt['beta2']))
			self.optimizers.append(self.optimizer_G)

			# schedulers
			if train_opt['lr_scheme'] == 'MultiStepLR':
				for optimizer in self.optimizers:
					self.schedulers.append(
						lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
														 restarts=train_opt['restarts'],
														 weights=train_opt['restart_weights'],
														 gamma=train_opt['lr_gamma'],
														 clear_state=train_opt['clear_state']))
			elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
				for optimizer in self.optimizers:
					self.schedulers.append(
						lr_scheduler.CosineAnnealingLR_Restart(
							optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
							restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
			else:
				raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

			self.log_dict = OrderedDict()

	def feed_data(self, data):
		self.real_H = data['GT']  # GT
		
		if "LQ" in data:
			self.ref_L = data['LQ']  # GT
		if self.opt['train']:
			self.real_H = self.real_H.to(self.device)
			# if "LQ" in data:
			#     self.ref_L = self.ref_L.to(self.device)
		print(self.real_H.size())
		self.real_H = self.real_H.transpose(1,2).reshape(-1,3,self.real_H.size(3),self.real_H.size(4))
		if "LQ" in data: 
			self.ref_L = self.ref_L.transpose(1,2).reshape(-1,3,self.ref_L.size(3),self.ref_L.size(4))
		else:
			if self.opt["distortion"] == "pytorch_bicubic":
				self.ref_L = F.upsample(self.real_H,scale_factor=(1/self.opt['scale'],1/self.opt['scale']),mode='area')
			if self.opt["distortion"] == "sr_bd":
				self.ref_L = Guassian_downsample(self.real_H.transpose(0,1),scale=2).transpose(0,1)
	   
	def gaussian_batch(self, dims):
		return torch.randn(tuple(dims)).to(self.device)

	def loss_forward(self, out, y):
		l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
		return l_forw_fit

	def loss_backward(self, x, y):
		# print("y",y.size())
		x_samples1 = self.netG(x=y, rev=True)
		# x_samples_image = x_samples[:, :3, :, :]
		l_back_rec1 = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples1)
		# l_back_rec2 = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples2)

		return l_back_rec1


	def optimize_parameters(self, step):
		# forward downscaling
		self.input = self.real_H
		self.output,LR_codec_recons,loss_c,distortion_loss,distribution_loss\
			,mimick_loss,img_bpp = self.netG(x=self.input,rev=False)
		
		loss_c = loss_c.mean()*self.train_opt['lambda_cond_prob']
		distortion_loss = distortion_loss.mean()*self.train_opt['lambda_distor_loss']
		distribution_loss = distribution_loss.mean()
		mimick_loss = mimick_loss.mean()*self.train_opt['lambda_mimick_loss']
		img_bpp = img_bpp.mean()
		LR_ref = self.ref_L.detach()
		l_forw_fit = self.loss_forward(self.output[:, :3, :, :], LR_ref)


		l_back_rec = self.loss_backward(self.real_H, LR_codec_recons)

		loss = (l_forw_fit+l_back_rec+loss_c+mimick_loss) * self.train_opt["loss_multiplier"]
		# loss = (l_forw_fit+l_back_rec+loss_c+mimick_loss)*128*10
		loss.backward()

		# gradient clipping
		if self.train_opt['gradient_clipping']:
			total_norm = nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
			if total_norm > self.train_opt['gradient_clipping']:
				print(("clipping gradient: {} with coef {}".format(total_norm, self.train_opt['gradient_clipping']/ total_norm)))
		self.optimizer_G.step()
		self.optimizer_G.zero_grad()


		# set log
		self.log_dict['l_forw_fit'] = l_forw_fit.item()
		self.log_dict['l_back_rec'] = l_back_rec.item()
		self.log_dict['loss_c'] = loss_c.item()
		self.log_dict['mimick_loss'] = mimick_loss.item()
		self.log_dict['distribution_loss'] = distribution_loss.item()
		# self.log_dict['img_distor_loss'] = img_distor_loss.item()
		self.log_dict['img_bpp'] = img_bpp.item()
		self.log_dict['loss'] = loss.item()

	def test(self):
		Lshape = self.ref_L.shape

		input_dim = Lshape[1]
		self.input = self.real_H
		

		zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

		gaussian_scale = 1
		if self.test_opt and self.test_opt['gaussian_scale'] != None:
			gaussian_scale = self.test_opt['gaussian_scale']

		self.netG.eval()
		with torch.no_grad():
			import time
			T1 = time.time()
			self.forw_L,LR_codec_recons,loss_c,distortion_loss,video_bpp\
			,mimick_loss,img_bpp= self.netG(x=self.input, rev=False)
			self.mimick_loss = mimick_loss
			self.img_bpp = img_bpp
			self.video_bpp = video_bpp
			self.video_distor_loss = distortion_loss
			T2 = time.time()
			print('down time %s ms' % ((T2 - T1)*1000))
			# self.forw_L = self.Quantization(self.forw_L)
			T1 = time.time()
			x_samples = self.netG(x=LR_codec_recons, rev=True)

			T2 = time.time()
			print('up time %s ms' % ((T2 - T1)*1000))
			self.fake_H = x_samples[:, :3, :, :]

			# l_forw_fit = self.loss_forward(self.output[:, :3, :, :], LR_ref)
			# print(self.real_H.device,  self.fake_H.device)
			# l_back_rec = self.Reconstruction_back(self.real_H, self.fake_H)
			# print("l_back_rec",l_back_rec)

		self.netG.train()

	def downscale(self, HR_img):
		self.netG.eval()
		with torch.no_grad():
			LR_img = self.netG(x=HR_img)[:, :3, :, :]
			LR_img = self.Quantization(self.forw_L)
		self.netG.train()

		return LR_img

	def upscale(self, LR_img, scale, gaussian_scale=1):
		Lshape = LR_img.shape
		zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
		y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

		self.netG.eval()
		with torch.no_grad():
			HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
		self.netG.train()

		return HR_img

	def get_current_log(self):
		return self.log_dict

	def get_current_metrics(self):
		out_dict = OrderedDict()
		out_dict['video_distor_loss'] = self.video_distor_loss.detach().float().cpu()
		out_dict['video_bpp'] = self.video_bpp.detach().float().cpu()
		out_dict['mimick_loss'] = self.mimick_loss.detach().float().cpu()
		out_dict['img_bpp'] = self.img_bpp.detach().float().cpu()
		return out_dict
	def get_current_visuals(self):
		out_dict = OrderedDict()
		
		ref_l = self.ref_L
		ref_l =  ref_l.reshape(-1,3,ref_l.size(2),ref_l.size(3))

		fake_h = self.fake_H
		print("fake_h",fake_h.size())
		fake_h =  fake_h.reshape(-1,3,fake_h.size(2),fake_h.size(3))

		forw_l = self.forw_L

		forw_l =  forw_l.reshape(-1,3,forw_l.size(2),forw_l.size(3))


		real_h = self.real_H
		real_h =  real_h.reshape(-1,3,real_h.size(2),real_h.size(3))

		# fake_h = fake_h[1:2]
		# real_h = real_h[1:2]

		out_dict['LR_ref'] = ref_l.detach().cpu()
		out_dict['SR'] = fake_h.detach().cpu()
		out_dict['LR'] = forw_l.detach().cpu()
		out_dict['GT'] = real_h.detach().cpu()
		return out_dict

	def print_network(self):
		s, n = self.get_network_description(self.netG)
		if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
			net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
											 self.netG.module.__class__.__name__)
		else:
			net_struc_str = '{}'.format(self.netG.__class__.__name__)
		if self.rank <= 0:
			logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
			logger.info(s)


	def load(self):
		load_path_G = self.opt['path']['pretrain_model_G']
		if load_path_G is not None:
			logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
			self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

	def save(self, iter_label):
		self.save_network(self.netG, 'G', iter_label)
