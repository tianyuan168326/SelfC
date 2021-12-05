import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.Subnet_constructor import subnet
import math
logger = logging.getLogger('base')


####################
# define networks   
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    model_type  = opt["model"]

    if model_type in ["IRN",'IRN_Contra_UP']:
        from models.modules.Inv_arch import InvRescaleNet
        netG = InvRescaleNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)
    if model_type in ["SelfC","SelfC_shell"]:
        from models.modules.SelfC_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    if model_type in ["SelfC_GMM","SelfC_SR",'SelfC_CUT',\
        'SelfC_CUTPixel','SelfC_CUT_sep',
        'SelfC_CUT_adav','SelfC_CUT_energy','SelfC_CUT_energy_patch',
        'SelfC_CUT_energy_dism','SelfC_CUT_energy_distortion',\
            'SelfC_CUTdownup','SelfC_CUT_sep_GAN', "SelfC_Contra_UP"]:
        from models.modules.SelfC_GMM_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    if model_type in ['SelfC_CUTdownup_noInv','SelfC_CUT_sep_noInv']:
        from models.modules.SelfC_GMM_arch_noinv import SelfCNoInvNet
        netG = SelfCNoInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    if "plain" in model_type:
        from models.modules.Plain_arch import SelfCNoInvNet
        netG = SelfCNoInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    
    if model_type in ["SelfC_VRN"]:
        from models.modules.SelfC_VRN_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    if model_type in ["SelfC_VRN_haar"]:
        from models.modules.SelfC_VRN_haar_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    if model_type in ["VRN_CUT_sep",'VRN_Contra_UP','VRN','VRN_Contra_UP_index',"VRN_Cross",'VRN_Contra_UP_video']:
        from models.modules.VRN_networks import Net
        netG = Net(opt, subnet(subnet_type, init), down_num)

    
    if model_type in ["SelfC_EBM"]:
        from models.modules.SelfC_EMB_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    if model_type in ["SelfC_GMM_prog"]:
        from models.modules.SelfC_GMM_prog_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    elif model_type in ["SelfC_GMM_Codec"]:
        from models.modules.SelfC_Codec_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type,\
             opt_net['block_num'], down_num,all_opt = opt)
    elif model_type in ["VRN_Codec"]:
        # print(opt)
        # exit(0)
        from models.modules.VRN_Codec_arch_inv import Net
        netG = Net(opt, subnet(subnet_type, init), down_num)
    elif model_type in ["SelfC_Noise"]:
        from models.modules.SelfC_Noise_Lcond_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type,\
             opt_net['block_num'], down_num,all_opt = opt)
    elif model_type in ["SR_Noise"]:
        from models.modules.Noise_SR_arch import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type,\
             opt_net['block_num'], down_num,all_opt = opt)
    elif model_type in ["Encoder_Shell"]:
        from models.modules.Encoder_Shell import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type,\
             opt_net['block_num'], down_num,all_opt = opt)
    elif model_type in ["SelfC_Imgcodec"]:
        from models.modules.SelfC_ImgCodec_arch_inv import SelfCInvNet
        netG = SelfCInvNet(opt_net,opt_net['in_nc'], opt_net['out_nc'],subnet_type, opt_net['block_num'], down_num)
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
