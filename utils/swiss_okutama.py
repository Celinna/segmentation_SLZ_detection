#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:58:46 2022

@author: rigi
"""
import os
import numpy as np
import cv2 as cv 
from PIL import Image


class SwissOkutama():
    def __init__(self):
        # Label to RGB color
        self.colormap = {"Background": [0, 0, 0], 
                        "Outdoor structures": [237, 237, 237],
                        "Buildings": [181, 0, 0],
                        "Paved ground": [135, 135, 135],
                        "Non-paved ground": [189, 107, 0],
                        "Train tracks": [128, 0, 128],
                        "Plants": [31, 123, 22],
                        "Wheeled vehicles": [6, 0, 130],
                        "Water": [0, 168, 255],
                        "People": [240, 255, 0]}

        # Converts from color index to RGB values
        self.idx2color = {k:v for k,v in enumerate(list(self.colormap.values()))}

        # Not all labels are used
        ignore_label = 255
        self.label_mapping = {0: ignore_label, 
                            1: 0, 2: 1, 
                            3: 2, 4: 3,
                            5: 4, 6: 5, 
                            7: 6, 8: 7, 
                            9: ignore_label}
        
        self.mean=[0.39313033, 0.48066333, 0.45113695] # for BGR channels
        self.std=[0.1179, 0.1003, 0.1139]
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    

    def category2mask(self, img):
        """ Convert a category image to color mask """
        if len(img) == 3:
            if img.shape[2] == 3:
                img = img[:, :, 0]

        mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')
        
        for category, mask_color in self.idx2color.items():
            locs = np.where(img == category)
            mask[locs] = mask_color
    
        return mask

    def save_pred(self, preds, sv_path, name, rgb=False):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            if rgb:
                mask_rgb = self.category2mask(pred)
                save_img = Image.fromarray(mask_rgb)
                
                save_img.save(os.path.join(sv_path, name +'.png'))
            else:
                save_img = Image.fromarray(pred)
                save_img.save(os.path.join(sv_path, name[i]+'.png'))