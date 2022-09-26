#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:20:56 2022

@author: rigi
"""
import argparse
import numpy as np
import os
from os import path
from glob import glob

fold1 = {
        'train': ['D','H','F','K','E','C','I'], 
        'val': ['L'],
        'test': ['G','A','B','J']     
        }

fold2 = {
        'train': ['E','C','I','L','G','A','B'], 
        'val': ['J'],
        'test': ['D','H','F','K']     
        }

fold3 = {
        'train': ['G','A','B','J','D','H','F'], 
        'val': ['K'],
        'test': ['E','C','I','L']     
        }

def main(args):   
    if args.fold == 1:
        my_dict = fold1
    elif args.fold == 2:
        my_dict = fold2
    elif args.fold == 3:
        my_dict = fold3
    else:
        print("Requested train/val/test combination does not exist...\nPlease use number between 1 and 3.")
    
    newpath = "./crossval" + str(args.fold)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    for key in my_dict.keys():
        print("Parsing {} data...".format(key))
        imglist = []
        lablist = []
        if key == "test":
            datadir = 'regions'
        else:
            datadir = args.datadir
            
        for region in my_dict[key]:
            print(region)
            imgs = sorted(glob(os.path.join(datadir, region, 'images',"*.png")))
            masks = sorted(glob(os.path.join(datadir, region, 'ground_truth', "*.png")))
            imglist.extend(imgs)
            lablist.extend(masks)
            # print(imgs) 
            
        for i in range(len(imglist)):
            with open(os.path.join(newpath, key + ".lst"), "a") as a_file:
                full = imglist[i] +"\t"+ lablist[i]
                a_file.write(full)
                a_file.write("\n")
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Swiss Okutama data list saver")
    parser.add_argument('--datadir', default='regions_aug', help='root directory where dataset is saved')
    parser.add_argument('--fold', type=int, help='which cross validation fold to generate image list')

    args = parser.parse_args()
    
    main(args)