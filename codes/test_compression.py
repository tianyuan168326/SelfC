import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import torch
#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)
is_save_image = opt["save_image"]
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
ave_video_distor_loss_all_ds = []
ave_video_bpp_all_ds = []
ave_img_distor_loss_all_ds = []
ave_img_bpp_all_ds = []

for test_loader in test_loaders:
    torch.cuda.empty_cache() 

    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    test_results['psnr_lr'] = []
    test_results['ssim_lr'] = []
    test_results['psnr_y_lr'] = []
    test_results['ssim_y_lr'] = []

    metric_results = OrderedDict()
    metric_results['video_distor_loss'] = []
    metric_results['video_bpp'] = []
    metric_results['mimick_loss'] = []
    metric_results['img_bpp'] = []

    

    for data in test_loader:
        model.feed_data(data)
        img_path = data['GT_path'][0]
        name_stomic = img_path.split("/")
        # print(img_path)
        img_name = name_stomic[-3]+"_" + name_stomic[-2] + "_" + name_stomic[-1]
        print(img_name)


        model.test()
        visuals = model.get_current_visuals()
        metrices = model.get_current_metrics()
        video_distor_loss = metrices["video_distor_loss"]
        video_bpp = metrices["video_bpp"]
        mimick_loss = metrices["video_distor_loss"]
        if "mimick_loss" in metrices:
            mimick_loss = metrices["mimick_loss"]
        img_bpp = metrices["img_bpp"]
        metric_results["video_distor_loss"]+=[video_distor_loss]
        metric_results["video_bpp"]+=[video_bpp]
        metric_results["mimick_loss"]+=[mimick_loss]
        metric_results["img_bpp"]+=[img_bpp]
        logger.info(
                    '{:20s} - video_distor_loss: {:.6f}; video_bpp: {:.6f} dB; mimick_loss: {:.6f}; img_bpp: {:.6f} dB.'.
                format(img_name, video_distor_loss.mean(), video_bpp.mean(), mimick_loss.mean(), img_bpp.mean()))

        # sr_img = util.tensor2img(visuals['SR'])  # uint8
        # srgt_img = util.tensor2img(visuals['GT'])  # uint8
        # lr_img = util.tensor2img(visuals['LR'])  # uint8
        # lrgt_img = util.tensor2img(visuals['LR_ref'])  # uint8

        sr_img = (visuals['SR'])
        srgt_img = (visuals['GT']) 
        lr_img = (visuals['LR'])  
        lrgt_img = (visuals['LR_ref']) 
        
        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.jpg')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.jpg')
        if is_save_image:
            util.save_img(util.tensor2img(sr_img), save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_GT.jpg')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_GT.jpg')
        if is_save_image:
            util.save_img(util.tensor2img(srgt_img), save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_LR.jpg')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_LR.jpg')
        if is_save_image:
            util.save_img(util.tensor2img(lr_img), save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_LR_ref.jpg')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_LR_ref.jpg')
        if is_save_image:
            util.save_img(util.tensor2img(lrgt_img), save_img_path)

        # calculate PSNR and SSIM
        # gt_img = util.tensor2img()
        gt_img = visuals['GT']

        # gt_img = gt_img / 255.
        # sr_img = sr_img / 255.

        # lr_img = lr_img / 255.
        # lrgt_img = lrgt_img / 255.
        # print(sr_img.size(),gt_img.size())

        cropped_sr_img = sr_img
        cropped_gt_img = gt_img
        # print(cropped_sr_img.size(),cropped_gt_img.size())
        # psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        # ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
        psnr = util.calculate_psnr(cropped_sr_img, cropped_gt_img)
        ssim = util.calculate_ms_ssim(cropped_sr_img, cropped_gt_img)
        test_results['psnr'] += psnr 
        test_results['ssim'] += ssim

        # PSNR and SSIM for LR
        # psnr_lr = util.calculate_psnr(lr_img * 255, lrgt_img * 255)
        # ssim_lr = util.calculate_ssim(lr_img * 255, lrgt_img * 255)
        psnr_lr = util.calculate_psnr(lr_img , lrgt_img )
        ssim_lr = util.calculate_ssim(lr_img , lrgt_img )
        test_results['psnr_lr'] += psnr_lr
        test_results['ssim_lr'] += ssim_lr

        
        # logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}. LR PSNR: {:.6f} dB; SSIM: {:.6f}.'\
        #     .format(img_name, psnr, ssim, psnr_lr, ssim_lr))
        # break
    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    ave_psnr_lr = sum(test_results['psnr_lr']) / len(test_results['psnr_lr'])
    ave_ssim_lr = sum(test_results['ssim_lr']) / len(test_results['ssim_lr'])
    
    ave_psnr_all_ds+=[ave_psnr]
    ave_ssim_all_ds +=[ave_ssim]
    ave_psnr_lr_all_ds+=[ave_psnr_lr]
    ave_ssim_lr_all_ds+=[ave_ssim_lr]

    logger.info(
            '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. LR psnr: {:.6f} db; ssim: {:.6f}.\n'.format(
            test_set_name, ave_psnr, ave_ssim, ave_psnr_lr, ave_ssim_lr))

    ave_video_distor_loss = sum(metric_results['video_distor_loss']) / len(metric_results['video_distor_loss'])
    ave_video_distor_loss = ave_video_distor_loss.mean()
    ave_video_bpp = sum(metric_results['video_bpp']) / len(metric_results['video_bpp'])
    ave_video_bpp = ave_video_bpp.mean()
    ave_img_distor_loss = sum(metric_results['mimick_loss']) / len(metric_results['mimick_loss'])
    ave_img_distor_loss = ave_img_distor_loss.mean()
    ave_img_bpp = sum(metric_results['img_bpp']) / len(metric_results['img_bpp'])
    ave_img_bpp = ave_img_bpp.mean()


    ave_video_distor_loss_all_ds +=[ave_video_distor_loss]
    ave_video_bpp_all_ds +=[ave_video_bpp]
    ave_img_distor_loss_all_ds +=[ave_img_distor_loss]
    ave_img_bpp_all_ds +=[ave_img_bpp]

    logger.info(
            '----Average Compression results for {}----\n\t ave_video_distor_loss: {:.6f}; ave_video_bpp: {:.6f}dB. ave_img_distor_loss: {:.6f}; ave_img_bpp: {:.6f}dB.\n'.format(
            test_set_name, ave_video_distor_loss, ave_video_bpp, ave_img_distor_loss, ave_img_bpp))

ave_psnr_all_ds = sum(ave_psnr_all_ds)/len(ave_psnr_all_ds)
ave_ssim_all_ds = sum(ave_ssim_all_ds)/len(ave_ssim_all_ds)
ave_psnr_lr_all_ds = sum(ave_psnr_lr_all_ds)/len(ave_psnr_lr_all_ds)
ave_ssim_lr_all_ds = sum(ave_ssim_lr_all_ds)/len(ave_ssim_lr_all_ds)
logger.info(
            '----Average PSNR/SSIM results for All dataset----\n\tpsnr: {:.6f} db; ssim: {:.6f}. LR psnr: {:.6f} db; ssim: {:.6f}.\n'.format(
             ave_psnr_all_ds, ave_ssim_all_ds, ave_psnr_lr_all_ds, ave_ssim_lr_all_ds))

ave_video_distor_loss_all_ds = sum(ave_video_distor_loss_all_ds)/len(ave_video_distor_loss_all_ds)
ave_video_bpp_all_ds = sum(ave_video_bpp_all_ds)/len(ave_video_bpp_all_ds)
ave_img_distor_loss_all_ds = sum(ave_img_distor_loss_all_ds)/len(ave_img_distor_loss_all_ds)
ave_img_bpp_all_ds = sum(ave_img_bpp_all_ds)/len(ave_img_bpp_all_ds)



logger.info(
            '----Average Compression results for All dataset----\n\t ave_video_distor_loss: {:.6f}; ave_video_bpp: {:.6f}dB. ave_img_distor_loss: {:.6f}; ave_img_bpp: {:.6f}dB.\n'.format(
            ave_video_distor_loss_all_ds, ave_video_bpp_all_ds, ave_img_distor_loss_all_ds, ave_img_bpp_all_ds))

