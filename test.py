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

from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

height = 10 # m


def preprocess(image, size, dataset):
    # give new shape in w, h
    img_size = image.shape
    print("Image size: {}, Input size: {}".format(img_size[0:2], size))
    
    if img_size[0:2] != size:
        image = cv.resize(image, size, interpolation=cv.INTER_AREA)
    
    
    image = image.astype(np.float32)[:, :, ::-1]
    # shape passed to model should be (1, 3, h, w)
    image = (image / 255.0)
    
    image -= dataset.mean
    image /= dataset.std
    
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    return image

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
    image_orig = cv.imread(filename, cv.IMREAD_COLOR)
    mask_gt = cv.imread(filename.replace('images', 'ground_truth'), cv.IMREAD_GRAYSCALE)
    
    # change mask gt 
    # mask_gt = cv.resize(mask_gt, test_size, interpolation=cv.INTER_NEAREST)
    
    # prepare data
    image = preprocess(image_orig, test_size, dataset)
    
    # start inference
    start = timeit.default_timer()
    pred = model.inference(image)
    end = timeit.default_timer()
    mask = np.asarray(pred.softmax(dim=1).argmax(dim=1).cpu().to(int))

    # landing zone detection
    mask = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
    mask = dataset.convert_label(mask, inverse=True)
    mask = np.squeeze(mask)
    # mask[mask_gt==0] = 0
    
    mask_gt = dataset.category2mask(mask_gt)
    mask_rgb = dataset.category2mask(mask)
    
    if PLOT_SINGLE:
        plt.imshow(cv.cvtColor(image_orig, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        plt.imshow(Image.fromarray(mask_gt))
        plt.axis('off')
        plt.show()
        
        plt.imshow(Image.fromarray(mask_rgb))
        plt.axis('off')
        plt.show()
        
    # Plot 
    if PLOT3:
        fig, axs = plt.subplots(1, 3, figsize=(15,15))
        fig.tight_layout() 
    
        # temp= Image.open(os.path.join(image_dir, 'images', file))
        axs[0].imshow(cv.cvtColor(image_orig, cv.COLOR_BGR2RGB))
        axs[1].imshow(Image.fromarray(mask_gt))
        axs[2].imshow(Image.fromarray(mask_rgb))
        for a in axs:
            a.axis('off')
        # plt.title("w32, " + filename)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
        plt.show()

    if LANDING:
        kernel = utils.get_kernel(height, margin=10) #adjust based on height above ground
        landingPoint = utils.get_landing_zone(mask, kernel)
    
    # save image   
    if SAVE:
        sv_path = os.path.join(path, 'test_results')
        print(sv_path)
        name =  os.path.splitext(os.path.basename(filename))[0]

        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        
        plt.imshow(Image.fromarray(mask_rgb))
        plt.axis('off')
        plt.savefig(os.path.join(sv_path, name), dpi=200, bbox_inches='tight', pad_inches = 0)
        plt.show()
    
    print('Seconds: {}'.format(end-start))


if __name__ == '__main__':
    # Settings
    SAVE = False
    REALTIME = True
    PLOT3 = False
    PLOT_SINGLE = True
    LANDING = False

    model_path = {'ddrnet23': '/home/rigi/segmentation/semantic-segmentation/output/ddrnet/fold2',
                  'ddrnet39': '/home/rigi/segmentation/semantic-segmentation/output/ddrnet39',
                  'bisenet': '/home/rigi/segmentation/semantic-segmentation/output/bisenet/fold2',
                  }
    model_dict= {'ddrnet23': 'DDRNet_DDRNet-23slim_SwissOkutama.pth',
                 'ddrnet39': 'DDRNet39_1080_SwissOkutama.pth',
                 'bisenet': 'BiSeNetv2_ResNet-18_SwissOkutama.pth'}
    
    # configure input image
    image_dir = '/home/rigi/segmentation/datasets/Okutama-Swiss-dataset/regions/J'
    
    filename = 'swiss_IMG_8776_1,1.png' # test 8748
    file = os.path.join(image_dir, 'images', filename)
    
    # configure model location
    if REALTIME:
        cfg_path = "./configs/ddrnet39.yaml"
        model = 'ddrnet39'
        path = model_path[model]
        model_state_file = os.path.join(model_path[model], model_dict[model])
        
    else:
        direc = ['seg_hrnet_w18_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_4_epoch100_fold2', 
                   'seg_hrnet_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold2', 
                   'seg_hrnet_w48_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold2']
        
        # path = '/home/rigi/segmentation/HRNet-Semantic-Segmentation/output/final_models'
        path = '/home/rigi/segmentation/HRNet-Semantic-Segmentation/output/final_models'
        cfg_path = "./seg_hrnet_w32_config.yaml"
        path = os.path.join(path, direc[1])
        model_state_file  = os.path.join(path, 'final_state.pth')
    
    print(model_state_file)
   
    print(file)
    main(file)
    # for file in glob.glob(image_dir + "/images/*.png"):
    #     print(file)
    #     main(file)