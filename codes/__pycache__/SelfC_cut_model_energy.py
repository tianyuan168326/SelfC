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
from models.modules.Quantization_softmax import Quantizer
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
from .CUT_energy_networks import *
from .patchnce import *
# torch.autograd.set_detect_anomaly(True)


def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))
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
        self.net_cut_g = MyResnetEncoder(3,3,ngf=64,n_blocks=12).to(self.device)
        self.net_cut_f1 = PatchSampleF1(use_mlp=True,nc= 256,gpu_ids=[self.device]).to(self.device)
        self.net_d = NLayerDiscriminator(3, 64, n_layers=3).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, \
                device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
            self.net_cut_g = DistributedDataParallel(self.net_cut_g, \
                device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
       
            self.net_cut_f1 = DistributedDataParallel(self.net_cut_f1, \
                device_ids=[torch.cuda.current_device()],find_unused_parameters=True)

            self.net_d = DistributedDataParallel(self.net_d, \
                device_ids=[torch.cuda.current_device()],find_unused_parameters=True,
                broadcast_buffers = False
                )
        else:
            self.netG = DataParallel(self.netG)
            self.net_cut_g = DataParallel(self.net_cut_g)
            self.net_cut_f = DataParallel(self.net_cut_f)
        
        # print network
        self.print_network()
        self.load()

        # self.Quantization = Quantizer(torch.linspace(0,1,256))
        self.Quantization = Quantization(is_clip=False)

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

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
            
            self.optimizer_net_cut_f1 = torch.optim.Adam(self.net_cut_f1.parameters(), lr=train_opt['lr_G'],
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            optim_param_not_emb = []
            optim_param_emb = []
            for k, v in self.net_cut_g.named_parameters():
                if v.requires_grad:
                    if 'ebm' in k or 'est' in k:
                        optim_param_emb.append(v)
                    else:
                        optim_param_not_emb.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))          
            self.optimizer_net_cut_g_ebm = torch.optim.Adam(optim_param_emb, lr=train_opt['lr_G']*2,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizer_net_cut_g_notebm = torch.optim.Adam(optim_param_not_emb, lr=train_opt['lr_G'],
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizer_D = torch.optim.Adam(self.net_d .parameters(), lr=train_opt['lr_G'], \
                betas=(train_opt['beta1'], train_opt['beta2']),
                )
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_net_cut_g_ebm)
            self.optimizers.append(self.optimizer_net_cut_g_notebm)
            self.optimizers.append(self.optimizer_net_cut_f1)
            self.optimizers.append(self.optimizer_D)

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


            self.criterionNCE = []
            # self.nce_layers = [0,5,10,15]
            self.nce_layers = [5,10,15]

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss2(opt).to(self.device))
            self.criterionGAN = GANLoss("lsgan").to(torch.cuda.current_device())
    def feed_data(self, data):
        self.real_H = data['GT']  # GT
        ##### padding extra images for small clip
        _1,_2,_3,_4,_5 = self.real_H.size()
        video_clip_len_train =  GlobalVar.get_Temporal_LEN()
        if (self.real_H.size(2) < video_clip_len_train):
            pad_num = video_clip_len_train  - self.real_H.size(2)
            pad_t = torch.zeros((_1,_2,pad_num,_4,_5))
            self.real_H = torch.cat([self.real_H,pad_t],dim=2)
        #####

        # if "LQ" in data:
        #     self.ref_L = data['LQ']  # GT
        if self.opt['train']:
            self.real_H = self.real_H.to(self.device)
        #     if "LQ" in data:
        #         self.ref_L = self.ref_L.to(self.device)
        self.real_H = self.real_H.transpose(1,2).reshape(-1,3,self.real_H.size(3),self.real_H.size(4))
        
        if "LQ" in data: 
            self.ref_L = self.ref_L.transpose(1,2).reshape(-1,3,self.ref_L.size(3),self.ref_L.size(4))
        else:
            if self.opt["distortion"] == "pytorch_bicubic":
                self.ref_L = F.upsample(self.real_H,scale_factor=(1/self.opt['scale'],1/self.opt['scale']),mode='area')
            if self.opt["distortion"] == "sr_bd":
                self.ref_L = Guassian_downsample(self.real_H.transpose(0,1)).transpose(0,1)
        
    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, fake_lr, real_h,ref_L):
        l_forw_fit =  self.Reconstruction_forw(fake_lr, ref_L)
        tgt = F.upsample(fake_lr,scale_factor= (4,4),mode="nearest")
        # tgt = fake_lr


        src = real_h

        n_layers = len(self.nce_layers)
        feat_q,_,_,_ = self.net_cut_g(tgt,self.nce_layers,cal_neg = False)
        self.num_patches = 512
        feat_k,energy_losses,coarse_est_loss_s,negs = self.net_cut_g(src,self.nce_layers,cal_neg=True)
       
        # print((negs[1] - feat_k[1].repeat(128,1,1)).abs().mean(),(negs[1] - feat_q[1].repeat(128,1,1)).abs().mean())

        feat_q_pool = self.net_cut_f1(feat_q, self.num_patches, None) ## b  c
        feat_k_pool = self.net_cut_f1(feat_k, self.num_patches, None) ## b  c
        negs_pool = self.net_cut_f1(negs, self.num_patches,None) ##n b c

        total_nce_loss = 0.0
        for f_q, f_k,neg, crit, nce_layer in zip(feat_q_pool, feat_k_pool,negs_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k,neg,tgt.size(0) ) * 1
            total_nce_loss += loss.mean()
        # print(energy_losses)
        # exit()
        energy_loss = sum(energy_losses)/len(energy_losses)
        coarse_est_los = sum(coarse_est_loss_s)/len(coarse_est_loss_s)
        return total_nce_loss / n_layers* self.train_opt['lambda_contra'] ,\
            l_forw_fit*self.train_opt['lambda_fit_forw'],\
            energy_loss,coarse_est_los
        # + loss_ph
        # return l_forw_fit
    def loss_backward(self, x, y):
        x_samples,_ = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        return l_back_rec

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def optimize_parameters(self, step):
        

        # forward downscaling
        self.input = self.real_H
        self.output,loss_c = self.netG(x=self.input,rev=False)
        loss_c = loss_c.mean()*self.train_opt['lambda_cond_prob']
        LR_ref = self.ref_L.detach()
        lr_before_quant = self.output[:, :3, :, :]
        # print(lr_before_quant.mean(),lr_before_quant.max(),lr_before_quant.min())
        l_penalty_output = torch.max((lr_before_quant-0.5)**2-0.25, \
            torch.zeros_like(lr_before_quant).cuda(lr_before_quant.device))[0]
        l_penalty_output = l_penalty_output.mean()*self.train_opt['lambda_penalty']
        LR = (self.Quantization((lr_before_quant)))
        l_quant = ((LR.clone().detach()-lr_before_quant)**2).mean()*self.train_opt['lambda_quant']
        l_contra,l_forw_fit,l_ennergy,coarse_est_loss = self.loss_forward(lr_before_quant, self.real_H,self.ref_L) 
 

        self.optimizer_G.zero_grad()
        self.optimizer_net_cut_f1.zero_grad()
        self.optimizer_net_cut_g_notebm.zero_grad()
        y_ = LR
        l_back_rec = self.loss_backward(self.real_H, y_)

        # self.set_requires_grad(self.net_d, False)
        # pred_fake = self.net_d(fake)
        # loss_G_GAN = self.criterionGAN(pred_fake, True).mean() *self.train_opt['lambda_lr_gan']
        # total loss
        # loss = l_forw_fit + l_back_rec +loss_c  +l_contra +loss_G_GAN
        
        loss = l_forw_fit + l_back_rec +loss_c  +l_contra +l_penalty_output 
        loss = loss*10
        loss = loss
        
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            total_norm = nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
            if total_norm > self.train_opt['gradient_clipping']:
                print(("clipping gradient: {} with coef {}".format(total_norm, self.train_opt['gradient_clipping']/ total_norm)))
        if self.train_opt['gradient_clipping']:
            total_norm = nn.utils.clip_grad_norm_(self.net_cut_g.parameters(), self.train_opt['gradient_clipping'])
            if total_norm > self.train_opt['gradient_clipping']:
                print(("clipping net_cut_g gradient: {} with coef {}".format(total_norm, self.train_opt['gradient_clipping']/ total_norm)))
        if self.train_opt['gradient_clipping']:
            total_norm = nn.utils.clip_grad_norm_(self.net_cut_f1.parameters(), self.train_opt['gradient_clipping'])
            if total_norm > self.train_opt['gradient_clipping']:
                print(("clipping net_cut_f1 gradient: {} with coef {}".format(total_norm, self.train_opt['gradient_clipping']/ total_norm)))
        self.optimizer_G.step()
        self.optimizer_net_cut_f1.step()
        self.optimizer_net_cut_g_notebm.step()

        self.optimizer_net_cut_g_ebm.zero_grad()
        coarse_est_loss = coarse_est_loss*0.1
        (l_ennergy+coarse_est_loss).backward()
        if self.train_opt['gradient_clipping']:
            clip_grad(self.net_cut_g.parameters(),self.optimizer_net_cut_g_ebm)
            # total_norm = nn.utils.clip_grad_norm_(self.net_cut_g.parameters(), 2000)
            # if total_norm > 2000:
            #     print(("clipping net_cut_g energy gradient: {} with coef {}".format(total_norm, 2000/ total_norm)))
        self.optimizer_net_cut_g_ebm.step()


        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_contra'] = l_contra.item()
        self.log_dict['l_quant'] = l_quant.item()
        self.log_dict['l_penalty_output'] = l_penalty_output.item()
        self.log_dict['l_ennergy'] = l_ennergy.item()
        self.log_dict['coarse_est_loss'] = coarse_est_loss.item()
        
        
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['loss_c'] = loss_c.item()
        self.log_dict['loss'] = loss.item()

        # self.log_dict['lr_gan_d'] = self.loss_D.item()
        # self.log_dict['lr_gan_g'] = loss_G_GAN.item()

        

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
            self.forw_L,_ = self.netG(x=self.input)
            T2 = time.time()
            # print('down time %s ms' % ((T2 - T1)*1000))

            self.forw_H = self.forw_L[:, 3:, :, :]
            self.forw_L = self.forw_L[:, :3, :, :]

            # print(self.forw_L .size())
            # exit()
            # x_soft, x_hard, symbols_hard= self.Quantization(self.forw_L)
            x_hard= self.Quantization(torch.clamp(self.forw_L,0,1))
            self.forw_L  = x_hard

            y = self.forw_L
            T1 = time.time()
            # flops, params = profile(self.netG.module, inputs=(y, True))
            # flops, params = clever_format([flops, params], "%.3f")
            # print("SELFC flops,params",flops,params)
            # time.sleep(2)
            # exit()
            x_samples,self.sample_H = self.netG(x=(y), rev=True)
            T2 = time.time()
            # print('up time %s ms' % ((T2 - T1)*1000))

            self.fake_H = x_samples[:, :3, :, :]

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
       
        # ref_l =  ref_l.reshape(-1,3,ref_l.size(2),ref_l.size(3))[0:1]
        # ref_l = ref_l[:,1]

        fake_h = self.fake_H
        # fake_h =  fake_h.reshape(-1,3,fake_h.size(2),fake_h.size(3))[0:1]
        # fake_h = fake_h[:,1]

        forw_l = self.forw_L
        # forw_l =  forw_l.reshape(-1,3,forw_l.size(2),forw_l.size(3))[0:1]
        # forw_l = forw_l[:,1]

        real_h = self.real_H
        # real_h =  real_h.reshape(-1,3,real_h.size(2),real_h.size(3))[0:1]
        # sample_h =  self.sample_H[3]
        # forw_H =  self.forw_H[3]
        sample_h =  self.sample_H
        forw_H =  self.forw_H
        # sample_h = F.pixel_shuffle(torch.cat((sample_h,sample_h[:,0:3]),dim=1),4)*100
        # forw_H = F.pixel_shuffle(torch.cat((forw_H,forw_H[:,0:3]),dim=1),4)*100

        # sample_h = ( (sample_h[:])).mean(dim=1).unsqueeze(1)*100
        # forw_H = ((forw_H[:])).mean(dim=1).unsqueeze(1)*100

        forw_H = ((forw_H[:,:]))
        # sample_h = ( (sample_h[:,:]))

        
        # forw_l = 0.4*forw_l.cpu()+0.6*ref_l.cpu()
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
        
        load_path_G = self.opt['path']['pretrain_model_net_cut_g']
        if load_path_G is not None:
            logger.info('Loading model for net_cut_g [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.net_cut_g, self.opt['path']['strict_load'])

        load_path_G = self.opt['path']['pretrain_model_net_cut_f1']
        if load_path_G is not None:
            logger.info('Loading model for pretrain_model_net_cut_f1 [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.net_cut_f1, self.opt['path']['strict_load'])

        load_path_G = self.opt['path']['pretrain_model_net_d']
        if load_path_G is not None:
            logger.info('Loading model for pretrain_model_net_d [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.net_d, self.opt['path']['strict_load'])
   
        # exit()
    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.net_cut_f1, 'net_cut_f1', iter_label)
        self.save_network(self.net_cut_g, 'net_cut_g', iter_label)
        self.save_network(self.net_d, 'net_d', iter_label)
