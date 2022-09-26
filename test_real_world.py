# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import os
import glob 

import timeit
import numpy as np
import cv2 as cv 
import yaml

from utils.swiss_okutama import SwissOkutama
from utils.seg_model import SegmentationModel
from utils import utils

import _init_paths
from config import config
from config import update_config

import matplotlib.pyplot as plt
from PIL import Image

# from skimage.io import imread, imsave
# from skimage import exposure
# from skimage.exposure import match_histograms


def preprocess(image, size, dataset, mean=None, std=None):
    # give new shape in w, h
    img_size = image.shape
    # print("Image size: {}, Input size: {}".format(img_size[0:2], size))
    
    # if img_size[0:2] != size:
    #     image = cv.resize(image, size, interpolation=cv.INTER_AREA)

    image = image.astype(np.float32)[:, :, ::-1]
    # shape passed to model should be (1, 3, h, w)
    
    image = (image / 255.0)
    
    if mean:
        # mean = [0.63801099, 0.6492692,  0.63265478]
        image -= mean
        # image += [0.1, 0.2, 0.1]
    
    else:
        image -= dataset.mean
        
    if std:
        image /= std
    else:
        image /= dataset.std
    
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    return image
          
def get_mean_std(img):
    mean = []
    std = []
    for i in range(img.shape[-1]):
        temp = img / 255.0
        mean.append(np.average(temp[:,:,i]))
        std.append(np.std(temp[:,:,i]))
        
    return mean, std
    

def main(filename):

    dataset = SwissOkutama() 
    
    if REALTIME:
        with open(cfg_path) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        test_size = (cfg['EVAL']['IMAGE_SIZE'][1], cfg['EVAL']['IMAGE_SIZE'][0]) #(H,W)
        # load model
        model = SegmentationModel(cfg, model_state_file, realtime=REALTIME, cuda_enabled=False)
        
    else:
        update_config(config, cfg_path)
        test_size = (config.TEST.IMAGE_SIZE[0], config.TEST.IMAGE_SIZE[1]) #(W,H)
        # load model
        model = SegmentationModel(config, model_state_file, realtime=REALTIME, cuda_enabled=False)

    # read image and mask
    img_orig = cv.imread(filename, cv.IMREAD_COLOR)

    # prepare data
   
    results = []
    new_images = []
    for i in range(3):
        img = img_orig.copy()
        mean, std = get_mean_std(img_orig)
        if i == 1:
            img[:,:,0] = (img[:,:,0]*0.8).clip(0,255)
            img[:,:,1] = (img[:,:,1]*1.0).clip(0,255)
            img[:,:,2] = (img[:,:,2]*1.1).clip(0,255)
            new_images.append(img)
            
        if i == 2:
            img[:,:,0] = (img[:,:,0]*0.62).clip(0,255)
            img[:,:,1] = (img[:,:,1]*0.95).clip(0,255)
            img[:,:,2] = (img[:,:,2]*0.95).clip(0,255)
            new_images.append(img)
        
        image = preprocess(img, test_size, dataset, mean=mean, std=std)
        
        # start inference
        start = timeit.default_timer()
        pred = model.inference(image)
        end = timeit.default_timer()
        mask = np.asarray(pred.softmax(dim=1).argmax(dim=1).cpu().to(int))
        
        # landing zone detection
        mask = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
        mask = dataset.convert_label(mask, inverse=True)
        mask = np.squeeze(mask)
        mask_rgb = dataset.category2mask(mask)
        
        results.append(mask_rgb)

    
    if PLOT_SINGLE:
        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        
        # plt.imshow(Image.fromarray(matched))
        # plt.axis('off')
        # plt.show()
        
        plt.imshow(Image.fromarray(mask_rgb))
        plt.axis('off')
        plt.show()
        
    # Plot 
    if PLOT3:
        fig, axs = plt.subplots(2, 3, figsize=(10,5))
        fig.tight_layout() 
    
        # temp= Image.open(os.path.join(image_dir, 'images', file))
        axs[0,0].imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
        axs[0,1].imshow(cv.cvtColor(new_images[0], cv.COLOR_BGR2RGB))
        axs[0,2].imshow(cv.cvtColor(new_images[1], cv.COLOR_BGR2RGB))
        axs[1,0].imshow(Image.fromarray(results[0]))
        axs[1,1].imshow(Image.fromarray(results[1]))
        axs[1,2].imshow(Image.fromarray(results[1]))
        
        axs[0,0].axis('off')
        axs[0,1].axis('off')
        axs[0,2].axis('off')
        axs[1,0].axis('off')
        axs[1,1].axis('off')
        axs[1,2].axis('off')
        # axs[3].imshow(Image.fromarray(mask_rgb))
        # for i, j in axs:
        #     a[i,j].axis('off')
        # plt.title("w32, " + filename)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.000, hspace=0.08)

        if SAVE:
            sv_path = '/home/rigi/thesis/real'
            name =  os.path.splitext(os.path.basename(filename))[0]

            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
           
            plt.imshow(Image.fromarray(mask_rgb))
            plt.axis('off')
            plt.savefig(os.path.join(sv_path, 'all_' + name), dpi=200, bbox_inches='tight', pad_inches = 0)
        plt.show()

    if LANDING:
        kernel = utils.get_kernel(height, margin=10) #adjust based on height above ground
        landingPoint = utils.get_landing_zone(mask, kernel)
    
    
    print('Seconds: {}'.format(end-start))


if __name__ == '__main__':
    # Settings
    SAVE = True
    REALTIME = False
    PLOT3 = True
    PLOT_SINGLE = False
    LANDING = False

    # configure input image
    image_dir = '/home/rigi/test_data/flights/orig'
    # image_dir = '/home/rigi/segmentation/datasets/Okutama-Swiss-dataset/regions/K/images'
    # filename = 'swiss_IMG_8745_1,0.png' # test 8748
    filename = 'vlcsnap-2022-09-07-22h01m43s656.png'
    # file = os.path.join(image_dir, 'images', filename)
    file = os.path.join(image_dir, filename)
    
    # configure model location
    if REALTIME:
        path = "/home/rigi/segmentation/semantic-segmentation/output/bisenet/fold2"
        model_name = 'BiSeNetv2_ResNet-18_SwissOkutama.pth'
        cfg_path = "./ddrnet39.yaml"
        model_state_file = os.path.join(path, model_name)
        
    else:
        direc = ['seg_hrnet_w18_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_4_epoch100_fold2', 
                   'seg_hrnet_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold2', 
                   'seg_hrnet_w48_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold2',
                   'seg_hrnet_w32_train_480x640_sgd_lr1e-2_wd5e-4_bs_12_epoch100_fold2_n']
        
        # path = '/home/rigi/segmentation/HRNet-Semantic-Segmentation/output/final_models'
        path = '/home/rigi/segmentation/HRNet-Semantic-Segmentation/output/final_models'
        cfg_path = "./seg_hrnet_w32_config.yaml"
        path = os.path.join(path, direc[-1])
        model_state_file  = os.path.join(path, 'best.pth')
    
    # print(model_state_file)
   
    # print(file)
    # main(file)
    print(image_dir)
    for file in glob.glob(image_dir + "/*.png"):
        print(file)
        main(file)
