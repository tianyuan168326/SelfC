import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model in ['SelfC',"SelfC_VRN","SelfC_GMM"]:
        from .SelfC_model import SelfCModel as M
    elif model in ["SelfC_GMM_Codec"]:
        from .SelfC_Codec_model import SelfCModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
