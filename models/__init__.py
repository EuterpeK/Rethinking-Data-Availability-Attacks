from .resnet import resnet18, resnet50
from .resnet import wrn34_10
from .vgg import vgg11_bn, vgg16_bn, vgg19_bn
from .densenet import densenet121
from .vit import ViT
from .swin import swin_t, swin_b, swin_s, swin_l


import torch
import numpy as np

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().cuda()
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().cuda()

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas