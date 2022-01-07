#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import logging
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

        
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    logger.info("INITIALIZE MODEL FOR FINETUNING")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    inf = model.fc.in_features
    model.fc = nn.Linear(inf, 133)
    
    return model
    
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"DEVICE: {device}")
    
    logger.info("LOADING MODEL WEIGHTS")
    model = net().to(device)
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model
