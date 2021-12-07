import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from global_var import GlobalVar
import os



class LQGTVIDDataset(data.Dataset):
	'''
	Read LQ (Low Quality, here is LR) and GT image pairs.
	If only GT image is provided, generate LQ image on-the-fly.
	The pair is ensured by 'sorted' function, so please check the name convention.
	'''

	def __init__(self, opt):
		super(LQGTVIDDataset, self).__init__()
		self.opt = opt
		self.data_type = self.opt['data_type']
		self.paths_LQ, self.paths_GT = None, None
		self.sizes_LQ, self.sizes_GT = None, None
		self.LQ_env, self.GT_env = None, None  # environment for lmdb
		# if not GlobalVar.get_Temporal_LEN() or not self.opt['video_len'] == GlobalVar.get_Temporal_LEN():
		# # print("self.opt['video_len']",self.opt['video_len'])
		#     GlobalVar.set_Temporal_LEN(self.opt['video_len'])
		# GlobalVar.set_Temporal_LEN(100)
		if not GlobalVar.get_Istrain():
			GlobalVar.set_Istrain(self.opt['phase'] == 'train')
		self.paths_GT, self.sizes_GT = util.get_vid_paths(self.data_type, opt['dataroot_GT'],opt['dataroot_list'],GlobalVar.get_Istrain())
		if self.opt['phase'] != 'train' and opt["sample_num"] and opt["sample_num"]>0:
			self.paths_GT = self.paths_GT[0:opt["sample_num"]]
		
		# self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
		# assert self.paths_GT, 'Error: GT path is empty.'
		# if self.paths_LQ and self.paths_GT:
		#     assert len(self.paths_LQ) == len(
		#         self.paths_GT
		#     ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
		#         len(self.paths_LQ), len(self.paths_GT))
		self.random_scale_list = [1]
		if opt["use_multi_scale"]:
			# self.random_scale_list = [0.6,0.8,1,1.2,1.4,1.6,1.8,2]
			self.random_scale_list = [0.5]
		video_len = self.opt['video_len']
		
		GlobalVar.set_Temporal_LEN(video_len) ## set global length 


	def _init_lmdb(self):
		# https://github.com/chainer/chainermn/issues/129
		self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
								meminit=False)
		self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
								meminit=False)
	def gen_aug_params(self):
		self.random_scale = random.choice(self.random_scale_list)
		self.hflip = self.opt['use_flip'] and random.random() < 0.5
		self.vflip = self.opt['use_rot'] and random.random() < 0.5
		self.rot90 = self.opt['use_rot'] and random.random() < 0.5
		self.need_aug = True
		
	def read_img(self,GT_path,scale,GT_size):
		resolution = None
		
		# modcrop in the validation / test phase
		# if self.opt['phase'] == 'train':
		#     img_GT = util.read_img1(None, GT_path, resolution)
		#     img_GT = util.modcrop(img_GT, 128)
		#     img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
		# else:
		#     img_GT = util.read_img(None, GT_path, resolution)
		# change color space if necessary
		img_GT = util.read_img1(None, GT_path, resolution)
		# img_GT = util.modcrop(img_GT, 4)
		img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
		

		# # randomly scale during training
		# if self.opt['phase'] == 'train':
		#     random_scale = self.random_scale
		#     H_s, W_s, _ = img_GT.shape

		#     def _mod(n, random_scale, scale, thres):
		#         rlt = int(n * random_scale)
		#         rlt = (rlt // scale) * scale
		#         return thres if rlt < thres else rlt

		#     H_s = _mod(H_s, random_scale, scale, GT_size)
		#     W_s = _mod(W_s, random_scale, scale, GT_size)
		#     img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
			

		_,H, W = img_GT.shape
		# using matlab imresize
		# img_LQ = util.imresize_np(img_GT, 1 / scale, True)
		# if img_LQ.ndim == 2:
		#     img_LQ = np.expand_dims(img_LQ, axis=2)

		if self.opt['phase'] == 'train':
			# if the image size is too small
			H, W, _ = img_GT.shape
			if H < GT_size or W < GT_size:
				img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
									interpolation=cv2.INTER_LINEAR)
				# using matlab imresize
				# img_LQ = util.imresize_np(img_GT, 1 / scale, True)
			   

			

			if self.need_aug:
				# randomly crop
				self.rnd_h = random.randint(0, max(0, H - GT_size))
				self.rnd_w = random.randint(0, max(0, W - GT_size))
			rnd_h = self.rnd_h 
			rnd_w = self.rnd_w 
			rnd_h_GT, rnd_w_GT = int(rnd_h), int(rnd_w )
			img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

			# augmentation - flip, rotate
			[img_GT] = util.augment([ img_GT], 
			self.hflip,
			self.vflip,
			self.rot90
			)

			if img_GT.shape[2] == 3:
				img_GT = img_GT[:, :, [2, 1, 0]]
			img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
		else:
			if self.opt["use_multi_scale"]:
				random_scale = 0.5
				H_s, W_s, _ = img_GT.shape

				def _mod(n, random_scale, scale):
					rlt = int(n * random_scale)
					rlt = (rlt // scale) * scale
					return rlt

				H_s = _mod(H_s, random_scale, 1)
				W_s = _mod(W_s, random_scale, 1)
				img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
			
			# img_GT = torch.from_numpy(img_GT)
			if img_GT.shape[2] == 3:
				img_GT = img_GT[:, :, [2, 1, 0]]


				
			img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
		self.need_aug  = False
		return img_GT,None
	def __getitem__(self, index):
		import random
		if self.data_type == 'lmdb':
			if (self.GT_env is None) or (self.LQ_env is None):
				self._init_lmdb()
		GT_path, LQ_path = None, None
		scale = self.opt['scale']
		GT_size = self.opt['GT_size']
		video_len = self.opt['video_len']
		# get GT image
		GT_path = self.paths_GT[index]
		vid_HQ = []
		vid_LQ = []
		self.gen_aug_params()
		# if not video_len  in [3,7,12] :
		#     print("only video length 3 7 supported")
		#     exit()
		# GT_paths = []
		
		# if video_len == 3:
		#     # GT_paths = [GT_path[random.randint(0,1)],GT_path[random.randint(2,3)],GT_path[random.randint(4,6)]]
		#     index1 = random.randint(0,4)
		#     index2 = random.randint(index1+1,5)
		#     index3 = random.randint(index2+1,6)
		# print(video_len)
		# exit()
		#     GT_paths = [GT_path[index1],GT_path[index2],GT_path[index3]]
		GT_path_len = len(GT_path)
		# print("GT_path_len",GT_path_len)
		# if self.opt['phase'] == 'test':
		#     GT_paths = GT_path
		if video_len == 5:
			if GT_path_len>5:
				#### train mode, randomly sampling
				if self.opt['phase'] == "train":
					index1 = random.randint(0,GT_path_len-5)
					index2 = random.randint(index1+1,GT_path_len-4)
					index3 = random.randint(index2+1,GT_path_len-3)
					index4 = random.randint(index3+1,GT_path_len-2)
					index5 = random.randint(index4+1,GT_path_len-1)
					GT_paths = [GT_path[index1],GT_path[index2],GT_path[index3],GT_path[index4],GT_path[index5]]
				#### test mode, continous sampling
				else:
					GT_paths = GT_path
			else:
				GT_paths = GT_path
		elif video_len == 3:
			if GT_path_len>3:
				index1 = random.randint(0,GT_path_len-3)
				index2 = random.randint(index1+1,GT_path_len-2)
				index3 = random.randint(index2+1,GT_path_len-1)
				GT_paths = [GT_path[index1],GT_path[index2],GT_path[index3]]
			else:
				GT_paths = GT_path
		elif video_len == 7:
			GT_paths = GT_path[0:video_len]
		elif video_len:
			GT_paths = GT_path[0:video_len]
		
		for img_path in GT_paths:
			# exit(0)
			img_GT,img_LQ = self.read_img(img_path,scale,GT_size)
			vid_HQ +=[img_GT]
			# vid_LQ +=[img_LQ]
		vid_HQ = torch.stack(vid_HQ,dim=1)
		# vid_LQ = torch.stack(vid_LQ,dim=1)
		# vid_HQ = vid_HQ.unsqueeze(0)
		# vid_LQ = vid_LQ.unsqueeze(0)
		# print(vid_HQ.size())


		
		return { 'GT': vid_HQ, 'LQ_path': GT_path[0], 'GT_path': GT_path[0]}

	def __len__(self):
		return len(self.paths_GT)
