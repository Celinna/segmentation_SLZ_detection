#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:03:22 2022

@author: rigi
"""

import torch
# import torch.nn as nn
from torch.nn import functional as F
# import torch.backends.cudnn as cudnn
import models

class SegmentationModel():
    def __init__(self, 
                 config, 
                 model_state_file,
                 realtime=False,
                 cuda_enabled=False):
        
        self.cuda_enabled = cuda_enabled
        self.model_state_file = model_state_file
        self.realtime = realtime
        
        self.load_model(config)
        
    def load_model(self, config):
        """
        Load the pytorch model
    
        Parameters
        ----------
        model_state_file : STR  - path to pytorch model in .pth
    
        Returns
        -------
        model : MODEL           - loaded model 
    ,
        """
        # if CUDA:
        #     # cudnn related setting
        #     cudnn.benchmark = config.CUDNN.BENCHMARK
        #     cudnn.deterministic = config.CUDNN.DETERMINISTIC
        #     cudnn.enabled = config.CUDNN.ENABLED
    
        if self.realtime:
            cfg = config
            # inference device cuda or cpu
            self.device = torch.device(cfg['DEVICE'])
    
            # initialize the model and load weights and send to device
            model = eval('models.'+config['MODEL']['NAME']+'.get_seg_model')()
            
            model.load_state_dict(torch.load(self.model_state_file, map_location='cpu'))
            # self.model = model.to(self.device)
            
        else:
        # build model
            if torch.__version__.startswith('1'):
                module = eval('models.'+ config.MODEL.NAME)
                module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
            model = eval('models.'+ config.MODEL.NAME +
                         '.get_seg_model')(config)
                
            pretrained_dict = torch.load(self.model_state_file, map_location=torch.device('cpu'))
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                                if k[6:] in model_dict.keys()}
        
                    
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        self.model = model
        
        # print(list(config.GPUS))
        # if CUDA:
        #     gpus = list(config.GPUS)
        #     model = nn.DataParallel(model, device_ids=gpus).cuda()
        
        # return model
        
        
    def inference(self, image):
        """
        Uses model to perform inference on input image

        Parameters
        ----------
        image : ARR          - RGB image in (1, 3, h, w) to perform segmentation
        model : MODEL        - loaded pytorch model from .pth file

        Returns
        -------
        pred : TENSOR - model prediction

        """
        self.model.eval()
        with torch.no_grad():
            image = torch.from_numpy(image)
            size = image.size()
            
            # make prediction
            pred = self.model(image)
            # pred = pred[0]
            print("Prediction size: {}".format(pred.size()))
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    input=pred, size=size[-2:],
                    mode='bilinear', align_corners=False
                )
                
            return pred.exp()          
                
        