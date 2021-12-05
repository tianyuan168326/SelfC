import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.loss import MotionFlowLoss
from models.modules.Quantization import Quantization
import utils.util as util
import torch.nn.functional as F
logger = logging.getLogger('base')
import numpy as np
import scipy
import scipy.ndimage
from .Guassian import Guassian_downsample
from thop import profile
# 增加可读性
from thop import clever_format
from global_var import *
import time
import models.modules.matlab_lr as matlab_lr

class SelfCModel(BaseModel):
    def __init__(self, opt):
        super(SelfCModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],find_unused_parameters=False)
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # self.motion_loss = MotionFlowLoss()
            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad and (not "opticFlow_Net" in k):
                    optim_params.append(v)
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
        ##### padding extra images for small clip
        _1,_2,_3,_4,_5 = self.real_H.size()
        video_clip_len_train =  GlobalVar.get_Temporal_LEN()
        # if (self.real_H.size(2) < video_clip_len_train):
        #     pad_num = video_clip_len_train  - self.real_H.size(2)
        #     pad_t = torch.zeros((_1,_2,pad_num,_4,_5))
        #     self.real_H = torch.cat([self.real_H,pad_t],dim=2)
        clip_length = self.real_H.size(2)
        if ( clip_length< video_clip_len_train):
            pad_num = video_clip_len_train  - clip_length
            pads = []
            for i in range(pad_num):
                pads += [self.real_H[:,:,-1,:,:]]
            pads = torch.stack(pads,dim=2)
            self.real_H = torch.cat([self.real_H,pads],dim=2)
        #####

        # if "LQ" in data:
        #     self.ref_L = data['LQ']  # GT
        if self.opt['train']:
            self.real_H = self.real_H.to(self.device)
        #     if "LQ" in data:
        #         self.ref_L = self.ref_L.to(self.device)
        self.real_H = self.real_H.transpose(1,2).reshape(-1,3,self.real_H.size(3),self.real_H.size(4))
        # img = util.tensor2img(self.real_H)
        # util.save_img(img, "tmp.jpg")
        # time.sleep(5000)
        # exit()
        if "LQ" in data: 
            self.ref_L = self.ref_L.transpose(1,2).reshape(-1,3,self.ref_L.size(3),self.ref_L.size(4))
        else:
            if self.opt["distortion"] == "pytorch_bicubic":
                self.ref_L = F.upsample(self.real_H,scale_factor=(1/self.opt['scale'],1/self.opt['scale']),mode='area')
            elif self.opt["distortion"] == "sr_bd":
                self.ref_L = Guassian_downsample(self.real_H.transpose(0,1)).transpose(0,1)
            elif self.opt["distortion"] == "matlab":
                self.ref_L = matlab_lr.imresize(self.real_H,scale=1/self.opt['scale'])
        return clip_length
    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_backward(self, x, y):
        x_samples,_ = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        return l_back_rec


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward downscaling
        self.input = self.real_H
        self.output,loss_c = self.netG(x=self.input,rev=False)
        loss_c = loss_c.mean()*self.train_opt['lambda_cond_prob']
        LR_ref = self.ref_L.detach()
        lr_before_quant = self.output[:, :3, :, :]
        l_forw_fit = self.loss_forward(lr_before_quant, LR_ref)
        # backward upscaling
        LR = self.Quantization(lr_before_quant)

        # gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        y_ = LR
        l_back_rec = self.loss_backward(self.real_H, y_)

        

        # total loss
        loss = l_forw_fit + l_back_rec +loss_c 
        loss = loss*144*144*3
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['loss_c'] = loss_c.item()
        self.log_dict['loss'] = loss.item()

    def test(self):

        self.input = self.real_H



        self.netG.eval()
        with torch.no_grad():
            import time
            T1 = time.time()
            bt, c, h, w = self.real_H.shape
            t = 7
            b = bt //t
            self.real_H = self.real_H.reshape(b,t,c,h,w)
            self.gop = 7
            n_gop = t // self.gop
            forw_L = []
            fake_H = []
            for i in range(n_gop + 1):
                if i == n_gop:
                    # calculate indices to pad last frame
                    indices = [i * self.gop + j for j in range(t % self.gop)]
                    for _ in range(self.gop - t % self.gop):
                        indices.append(t - 1)
                    self.input = self.real_H[:, indices]
                else:
                    self.input = self.real_H[:, i * self.gop:(i + 1) * self.gop]
                _b,_t,_c,_h,_w = self.input.shape
                self.forw_L,_ = self.netG(x=self.input.reshape(_b*_t,_c,_h,_w))
                T2 = time.time()
                # print('down time %s ms' % ((T2 - T1)*1000))

                self.forw_H = self.forw_L[:, 3:, :, :]
                self.forw_L = self.forw_L[:, :3, :, :]

                # print(self.forw_L .size())
                # exit()
                self.forw_L = self.Quantization(self.forw_L)
                y = self.forw_L
                T1 = time.time()
                # flops, params = profile(self.netG.module, inputs=(y, True))
                # flops, params = clever_format([flops, params], "%.3f")
                # print("SELFC flops,params",flops,params)
                # time.sleep(2)
                # exit()
                x_samples,self.sample_H = self.netG(x=y, rev=True)
                T2 = time.time()
                # print('up time %s ms' % ((T2 - T1)*1000))
                self.fake_H = x_samples[:, :3, :, :]
                self.forw_L = self.forw_L.reshape(b,_t,c,h//4,w//4)
                self.fake_H = self.fake_H.reshape(b,_t,c,h,w)
                if i == n_gop:
                        for j in range(t % self.gop):
                            forw_L.append(self.forw_L[:, j])
                            fake_H.append(self.fake_H[:, j])
                else:
                    for j in range(self.gop):
                        forw_L.append(self.forw_L[:, j])
                        fake_H.append(self.fake_H[:, j])
        self.fake_H = torch.stack(fake_H, dim=1)
        self.forw_L = torch.stack(forw_L, dim=1)
        b,t,c,h,w = self.fake_H.size()
        self.fake_H = self.fake_H.reshape(b*t,c,h,w)
        b,t,c,h,w = self.forw_L.size()
        self.forw_L = self.forw_L.reshape(b*t,c,h,w)
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

    def get_current_visuals(self):
        out_dict = OrderedDict()
        
        ref_l = self.ref_L
        

        fake_h = self.fake_H

        forw_l = self.forw_L

        real_h = self.real_H

        b,t,c,h,w = real_h.size()
        real_h = real_h.reshape(b*t,c,h,w)
        sample_h =  self.sample_H
        forw_H =  self.forw_H
        forw_H = ((forw_H[:,:]))

        out_dict['SR'] = fake_h.detach()
        out_dict['LR_ref'] = ref_l.detach()
        out_dict['LR'] = forw_l.detach()
        
        # out_dict['LR_diff'] = (forw_l.cpu()-ref_l.cpu()).detach()
        # out_dict['HR_diff'] = (real_h.cpu()-F.upsample(forw_l,scale_factor=(4,4)).cpu()).detach()
        out_dict['GT'] = real_h.detach()
        # out_dict['sample_h'] = sample_h.detach()
        out_dict['forw_H'] = forw_H.detach()
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
