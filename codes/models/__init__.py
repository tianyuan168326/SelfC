import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'IRN':
        from .IRN_model import IRNModel as M
    elif model in ['SelfC',"SelfC_VRN","SelfC_VRN_haar","SelfC_GMM","SelfC_GMM_prog",'VRN','plain']:
        from .SelfC_model import SelfCModel as M
    elif model in ['SelfC_EBM']:
        from .SelfC_EMB_model import SelfCModel as M
    elif model in ['SelfC_CUT']:
        from .SelfC_cut_model import SelfCModel as M
    elif model in ['SelfC_CUT_sep','SelfC_CUT_sep_noInv','VRN_CUT_sep','VRN_Contra_UP_video_plain']:
        from .SelfC_cutsep_model import SelfCModel as M
    elif model in ['VRN_Contra_UP', 'SelfC_Contra_UP','IRN_Contra_UP']:
        from .contrast_generator.SelfC_contra_up_model import SelfCModel as M
    elif model in ['VRN_Contra_UP_video']:
        from .contrast_generator.SelfC_contra_up_video_model import SelfCModel as M
    
    elif model in ['VRN_Cross']:
        from .contrast_generator.SelfC_cross_model import SelfCModel as M

    
    
    elif model in ['VRN_Contra_UP_index']:
        from .contrast_generator.SelfC_contra_up_index_model import SelfCModel as M
            
    elif model in ['SelfC_CUT_sep_GAN']:
        from .SelfC_cutsep_gan_model import SelfCModel as M
    elif model in ['SelfC_CUTdownup_noInv','SelfC_CUTdownup']:
        from .SelfC_cutdownup_model import SelfCModel as M
    
    elif model in ['SelfC_CUTPixel']:
        from .SelfC_pixel_model import SelfCModel as M
    
    elif model in ['SelfC_CUT_adav']:
        from .SelfC_cut_adav_model import SelfCModel as M
    
    elif model in ['SelfC_CUT_energy']:
        from .SelfC_cut_model_energy import SelfCModel as M
    elif model in ['SelfC_CUT_energy_dism']:
        from .SelfC_cut_model_energy_dism import SelfCModel as M
    elif model in ['SelfC_CUT_energy_distortion']:
        from .SelfC_cut_model_energy_distortion import SelfCModel as M
    elif model in ['SelfC_CUT_energy_patch']:
        from .SelfC_cut_model_energy_patch import SelfCModel as M 

       
    elif model in ['SelfC_v1']:
        from .SelfC_v1_model import SelfCModel as M
    elif model in ['SelfC_v1_plain']:
        from .SelfC_v1_model import SelfCModel as M
    
    elif model == "SelfC_Noise":
        from .SelfC_Noise_model import SelfCModel as M
    elif model in ["SelfC_GMM_Codec", 'VRN_Codec']:
        from .SelfC_Codec_model import SelfCModel as M
    elif model == "SR_Noise":
        from .SelfC_Noise_model import SelfCModel as M
    elif model == "Encoder_Shell":
        from .SelfC_Noise_model import SelfCModel as M
    elif model == "SelfC_SR":
        from .SelfC_SR_model import SelfCModel as M

    elif model == 'SelfC_DVCv2_Imgcodec_cross':
        from .SelfC_DVCv2_imgcodec_cross_model import SelfCModel as M
    elif model == 'SelfC_DVCv2_Imgcodec_res':
        from .SelfC_DVCv2_imgcodec_res_model import SelfCModel as M
    elif model == 'SelfC_DVCv2_Imgcodec_res1':
        from .SelfC_DVCv2_imgcodec_res1_model import SelfCModel as M
    elif model == 'SelfC_DVCv2_Imgcodec_gres':
        from .SelfC_DVCv2_imgcodec_gres_model import SelfCModel as M
    elif model == 'SelfC_DVCv2_Imgcodec_res1_sbpp':
        from .SelfC_DVCv2_imgcodec_res1_sbpp_model import SelfCModel as M
    elif model in ['SelfC_PretrainImgcodec_unshared','SelfC_PretrainImgcodec']:
        from .SelfC_pretrainimgcodec_model import SelfCModel as M
    elif model in ['SelfC_PretrainImgcodec_resquant']:
        from .SelfC_pretrainimgcodec_resquant_model import SelfCModel as M
    elif model in ['SelfC_PretrainImgcodec_Lcond']:
        from .SelfC_pretrainimgcodec_model import SelfCModel as M

    
    elif model == 'SelfC_Imgcodec':
        from .SelfC_Imgcodec_model import SelfCModel as M
        
    elif model == 'SelfC_shell':
        from .SelfC_shell_model import SelfCModel as M
    elif model == 'IRN+':
        from .IRNp_model import IRNpModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
