import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import os
import numpy as np
import options.options as option
import utils.util as util
# import utils.calculate_PSNR_SSIM as Metric_Calculator
from data.util import bgr2ycbcr,rgb_to_ycbcr
from data import create_dataset, create_dataloader
from models import create_model

calculate_psnr = util.calculate_psnr
calculate_ssim = util.calculate_ssim
#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
	(path for key, path in opt['path'].items()
	 if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
				  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
	test_set = create_dataset(dataset_opt)
	test_loader = create_dataloader(test_set, dataset_opt)
	logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
	test_loaders.append(test_loader)

model = create_model(opt)


ave_psnr_all_ds = []
ave_ssim_all_ds = []
ave_psnr_lr_all_ds = []
ave_ssim_lr_all_ds = []

ave_psnry_all_ds = []
ave_ssimy_all_ds = []
ave_psnry_lr_all_ds = []
ave_ssimy_lr_all_ds = []
def tensor_to_number(xs):
	l = []
	for i in xs:
		if type(i) == int or type(i) == float:
			l += [i]
		else:
			l += [i.item()]
	return l
img_ext = ".jpg"
def avg_list(l):
	if len(l) == 0:
		return 0
	return sum(l)/len(l)
import pickle
def cal_metric(val_loader,val_ds_name,model,opt,dataset_dir):
	avg_psnr = []
	avg_psnr_y = []
	avg_lr_psnr = []
	avg_lr_psnr_y = []

	avg_ssim = []
	avg_ssim_y = []
	avg_lr_ssim = []
	avg_lr_ssim_y = []
	idx = 0
	meta_metric_info = {}
	for val_data in val_loader:
		idx += 1
		# if idx == 3:
		# 	break

		
		print("testing progress {}/{}".format(idx*val_loader.batch_size, len(val_loader.dataset)))
		clip_length = model.feed_data(val_data)
		
		model.test()

		visuals = model.get_current_visuals()
		sr_img = (visuals['SR'])
		gt_img = (visuals['GT'])
		lr_img = (visuals['LR'])  
		lrgt_img = (visuals['LR_ref'])

		### save visualization
		bt = sr_img.size(0)
		video_len = 7
		b = bt//video_len
		sr_img_vis = sr_img.reshape(b,video_len,*sr_img.size()[-3:])
		gt_img_vis = gt_img.reshape(b,video_len,*gt_img.size()[-3:])
		lr_img_vis = lr_img.reshape(b,video_len,*lr_img.size()[-3:])
		lrgt_img_vis = lrgt_img.reshape(b,video_len,*lrgt_img.size()[-3:])
	   
		

		# calculate PSNR
		# avg_psnr += util.calculate_psnr(sr_img, gt_img)
		# avg_ssim += util.calculate_ssim(sr_img, gt_img)
		## cal Y channel
		sr_img_y = rgb_to_ycbcr(sr_img) 
		gt_img_y = rgb_to_ycbcr(gt_img) 
		batch_sr_psnr = util.calculate_psnr(sr_img_y, gt_img_y)
		avg_psnr_y+= [avg_list(batch_sr_psnr)]
		avg_ssim_y+= [avg_list(util.calculate_ssim(sr_img_y, gt_img_y))]

		# calculate LR PSNR
		# avg_lr_psnr += util.calculate_psnr(lr_img, lrgt_img)
		# avg_lr_ssim += util.calculate_ssim(lr_img, lrgt_img)
		## cal LR Y channel
		lr_y = rgb_to_ycbcr(lr_img) 
		lrgt_y = rgb_to_ycbcr(lrgt_img) 
		batch_lr_psnr = util.calculate_psnr(lr_y, lrgt_y)
		avg_lr_psnr_y += [avg_list(batch_lr_psnr)]
		avg_lr_ssim_y += [avg_list(util.calculate_ssim(lr_y, lrgt_y))]
		video_name_list = val_data['LQ_path']
		for b_i in range(b):
			video_name = os.path.splitext("_".join(video_name_list[b_i ].split("/")[-3:]))[0]
			for t_i in range(video_len):
				sr_1im = sr_img_vis[b_i,t_i]
				gt_1im = gt_img_vis[b_i,t_i]
				lr_1im = lr_img_vis[b_i,t_i]
				lrgt_1im = lrgt_img_vis[b_i,t_i]
				sr_psnr = batch_sr_psnr[b_i*video_len + t_i]
				lr_psnr = batch_lr_psnr[b_i*video_len + t_i]
				frame_path = os.path.join(dataset_dir,video_name+"_{}th".format(t_i))
				# print(frame_path)
				util.save_img(util.tensor2img(sr_1im), frame_path+"_sr"+img_ext)
				util.save_img(util.tensor2img(gt_1im), frame_path+"_gt"+img_ext)
				util.save_img(util.tensor2img(lr_1im), frame_path+"_lr"+img_ext)
				util.save_img(util.tensor2img(lrgt_1im), frame_path+"_lrgt"+img_ext)

				meta_metric_info[frame_path] = [sr_psnr,lr_psnr]
	avg_psnr = avg_list(avg_psnr)
	avg_psnr_y = avg_list(avg_psnr_y) 
	avg_lr_psnr = avg_list(avg_lr_psnr)
	avg_lr_psnr_y = avg_list(avg_lr_psnr_y)

	avg_ssim = avg_list(avg_ssim)
	avg_ssim_y = avg_list(avg_ssim_y)
	avg_lr_ssim = avg_list(avg_lr_ssim)
	avg_lr_ssim_y = avg_list(avg_lr_ssim_y)
	with open(dataset_dir+ "meta_info.pkl",'wb') as f:
		pickle.dump(meta_metric_info,f)
	return avg_psnr,avg_psnr_y,avg_lr_psnr,avg_lr_psnr_y, avg_ssim,avg_ssim_y,avg_lr_ssim,avg_lr_ssim_y

test_results = {}
test_results['psnr'] = []
test_results['ssim'] = []
test_results['psnr_y'] = []
test_results['ssim_y'] = []

test_results['psnr_lr'] = []
test_results['ssim_lr'] = []
test_results['psnr_y_lr'] = []
test_results['ssim_y_lr'] = []
for test_loader in test_loaders:
	test_set_name = test_loader.dataset.opt['name']
	logger.info('\nTesting [{:s}]...'.format(test_set_name))
	test_start_time = time.time()
	dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
	util.mkdir(dataset_dir)
	
	avg_psnr,avg_psnr_y,avg_lr_psnr,avg_lr_psnr_y,\
	avg_ssim,avg_ssim_y,avg_lr_ssim,avg_lr_ssim_y,\
	= \
	cal_metric(test_loader,test_set_name,model,opt,dataset_dir)
	logger.info(" HR results for {}, PSNR {:.6f}dB, SSIM {:.6f}".format(
	test_set_name,(avg_psnr_y), (avg_ssim_y)
	))
	logger.info(" LR results for {}, PSNR {:.6f}dB, SSIM {:.6f}".format(
	test_set_name,(avg_lr_psnr_y), (avg_lr_ssim_y)
	))
	test_results['psnr'] += [avg_psnr]
	test_results['ssim'] += [avg_ssim]
	test_results['psnr_y'] +=[avg_psnr_y]
	test_results['ssim_y'] += [avg_ssim_y]

	test_results['psnr_lr'] += [avg_lr_psnr]
	test_results['ssim_lr'] += [avg_lr_ssim]
	test_results['psnr_y_lr'] += [avg_lr_psnr_y]
	test_results['ssim_y_lr'] += [avg_lr_ssim_y]
	

logger.info("Averaged HR results for all datasets, PSNR {:.6f}dB, SSIM {:.6f}".format(
	avg_list(test_results['psnr_y']), avg_list(test_results['ssim_y'])
))

logger.info("Averaged LR results for all datasets, PSNR {:.6f}dB, SSIM {:.6f}".format(
	avg_list(test_results['psnr_y_lr']), avg_list(test_results['ssim_y_lr'])
))