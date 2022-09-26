#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:01:17 2022

@author: rigi
"""
import os
import numpy as np
import cv2 as cv 

import matplotlib.pyplot as plt


# LUT for erosion size:
# Keys: height (m)
# Vals: pixels per meter
height2pix = {1: 397.65, 2: 198.82, 3: 132.55, 4: 99.41, 5: 79.53, 
              6: 66.27, 7: 56.81, 8: 49.71, 9: 44.18, 10: 39.76, 
              11: 36.15, 12: 33.14, 13: 30.59, 14: 28.4, 15: 26.51, 
              16: 24.85, 17: 23.39, 18: 22.09, 19: 20.93, 20: 19.88, 
              21: 18.94, 22: 18.07, 23: 17.29, 24: 16.57, 25: 15.91, 
              26: 15.29, 27: 14.73, 28: 14.2, 29: 13.71, 30: 13.25, 
              31: 12.83, 32: 12.43, 33: 12.05, 34: 11.7, 35: 11.36, 
              36: 11.05, 37: 10.75, 38: 10.46, 39: 10.2, 40: 9.94, 
              41: 9.7, 42: 9.47, 43: 9.25, 44: 9.04, 45: 8.84, 
              46: 8.64, 47: 8.46, 48: 8.28, 49: 8.12, 50: 7.95, 
              51: 7.8, 52: 7.65, 53: 7.5, 54: 7.36, 55: 7.23, 
              56: 7.1, 57: 6.98, 58: 6.86, 59: 6.74, 60: 6.63, 
              61: 6.52, 62: 6.41, 63: 6.31, 64: 6.21, 65: 6.12, 
              66: 6.02, 67: 5.94, 68: 5.85, 69: 5.76, 70: 5.68, 
              71: 5.6, 72: 5.52, 73: 5.45, 74: 5.37, 75: 5.3, 
              76: 5.23, 77: 5.16, 78: 5.1, 79: 5.03, 80: 4.97, 
              81: 4.91, 82: 4.85, 83: 4.79, 84: 4.73, 85: 4.68, 
              86: 4.62, 87: 4.57, 88: 4.52, 89: 4.47, 90: 4.42, 
              91: 4.37, 92: 4.32, 93: 4.28, 94: 4.23, 95: 4.19, 
              96: 4.14, 97: 4.1, 98: 4.06, 99: 4.02, 100: 3.98}


def get_landing_zone(mask, kernel, thresh=0.30, show=False):
    """
    Get landing zone mark

    Parameters
    ----------
    mask : UINT8 ARR    - output grescale mask from segmentation with classes in 2D
    kernel : OTHER      - opencv structuring element as the kernal used for erosion

    Returns
    -------
    mask : UINT8 ARR    - binary mask with landing zone as foreground

    """
    size = mask.shape
    
    masks = {}
    masks['non-paved ground'] = np.zeros(size)
    masks['non-paved ground'][mask == 4] = 1
    
    
    masks['plants'] = np.zeros(size)
    masks['plants'] [mask == 6] = 1
    
    masks['water'] = np.zeros(size)
    masks['water'][mask == 8] = 1

    zone = None
    count = 0 
    for k, v in masks.items():
        
       
        masks[k] = cv.erode(v, kernel)
        plt.imshow(masks[k] , cmap='gray', vmin=0, vmax=1)
        # plt.title(k, fontsize=20)
        plt.axis('off')
        plt.show()
       
        area = np.sum(masks[k])
        if area > count:
            count = area
            zone = k
            
    if zone is not None:
        total_pxs = size[0] * size[1]
        
        if show:
            plt.imshow(masks[zone], cmap='gray', vmin=0, vmax=1)
            plt.show()
        
        if (count / total_pxs) >= thresh:
            print('Detected landing zone in {}!'.format(zone))
            
            # Get landing location
            landingPoint = get_location(masks[zone])
            return landingPoint
        
        else:
            print('No suitable landing zone available.')
            return None
    
    return mask
   
    

def get_location(mask):
    """
    Get landing point 

    Parameters
    ----------
    mask : TYPE
        DESCRIPTION.

    Returns
    -------
    landingPoint : TYPE
        DESCRIPTION.

    """
    row, col = mask.shape
    
    # Assuming target point centered on image
    center = (np.array((col, row))/2).astype('uint8')
    
    # Find white pixel closest to center starting from the center region
    landablePoints = np.squeeze(cv.findNonZero(mask))
    distances = np.sum((landablePoints - center)**2, axis=1)
    idx = np.argmin(distances)
    landingPoint = landablePoints[idx]
    
    return landingPoint
    

def get_kernel(height, margin, const=4):
    """
    Get erosion kernel

    Parameters
    ----------
    height : FLOAT  - distance to ground from range sensor (m)
    margin : INT    - real world landing margin in m

    Returns
    -------
    kernel : OTHER  - Opencv square structing element

    """
    h = round(height) # round height to nearest int
    
    if h > 100:    
        size = margin * const
        print("Height given is too high!")
    else:
        pixPerMeter = height2pix[h]
        size = round(margin * pixPerMeter)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size)) # use square element
    
    return kernel
