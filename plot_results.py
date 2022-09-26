# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import glob 

import numpy as np
import cv2 as cv 

from utils.swiss_okutama import SwissOkutama

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

       
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 8})
     
def main(filename):
    dataset = SwissOkutama()
    
    # read image and mask
    image = Image.open(os.path.join(image_dir, 'images', filename))
    gt = cv.imread(os.path.join(image_dir, 'ground_truth', filename), cv.IMREAD_GRAYSCALE)
    
    size = image.size
    print(size)
    out = []
    out.append(image)
    
    
    # Read predictions
    for item in model_list: 
        result_dir  = os.path.dirname(item)
        
        if 'HRNet' in item:
            pred = cv.imread(os.path.join(result_dir, 'test_results', filename), cv.IMREAD_GRAYSCALE)
            pred[gt==0] = 0
            pred = dataset.category2mask(pred)
            pred = Image.fromarray(pred)
        
        else:
            pred = Image.open(os.path.join(result_dir, 'test_results', filename))
        
        pred = pred.resize(size)
        # print("Image size: {}, Pred size: {}".format(image.size, pred.size))
        out.append(pred)
    
    out.insert(4, dataset.category2mask(gt))
    
    # Plot 
    print('Plotting...')
    
    fig, axs = plt.subplots(2, 4, dpi=80, constrained_layout=True, figsize=(10,4))
    # fig, axs = plt.subplots(2, 4, dpi=80, constrained_layout=True, figsize=(10,3))
    # fig.tight_layout() 
    text = ['IMG', 'HR-W18', 'HR-W32', 'HR-W48', 'GT',  'DDR-23', 'DDR-39', 'BI']
    # text = ['IMG', 'GT', '', '', '', '', '', '']
    j = 0
    k = 0
    for i in range(len(out)):
        if i < 4:
            axs[0,j].imshow(out[i])
            axs[0,j].axis('off')
            # axs[0,j].get_yaxis().set_visible(False)
            # text_box = AnchoredText(text[i], frameon=False, loc=4, pad=0.005)
            # plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            # axs[0,j].add_artist(text_box)
            # axs[0,j].set_title(text[i])
            j += 1
        else:
            axs[1,k].imshow(out[i])
            axs[1,k].axis('off')
            # text_box = AnchoredText(text[i], frameon=False, loc=4, pad=0.005)
            # plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            # axs[1,k].add_artist(text_box)
            k += 1
            
    # plt.title("w32, " + filename)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.06, hspace=0.001)
    if sv_pred:
        plt.savefig(os.path.join(sv_path, 'result' + filename), dpi=200, bbox_inches='tight', pad_inches = 0)
    plt.show()


def get_file_list():
    hrnet_dir = '/home/rigi/segmentation/HRNet-Semantic-Segmentation/output/final_models'
    hrnet_direc = ['seg_hrnet_w18_train_768x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch100_fold2_1', 
              'seg_hrnet_w32_train_768x1024_sgd_lr1e-2_wd5e-4_bs_8_epoch100_fold2_1', 
              'seg_hrnet_w48_train_768x1024_sgd_lr1e-2_wd5e-4_bs_4_epoch100_fold2_1',
              # 'seg_hrnet_ocr_w32_train_1080x1920_sgd_lr1e-2_wd5e-4_bs_2_epoch100_fold2'
              ]
    model_path = {'ddrnet23': '/home/rigi/segmentation/semantic-segmentation/output/ddrnet/fold2',
                  'ddrnet39': '/home/rigi/segmentation/semantic-segmentation/output/ddrnet39',
                  'bisenet': '/home/rigi/segmentation/semantic-segmentation/output/bisenet/fold2',
                  }
    model_dict= {'ddrnet23': 'DDRNet_DDRNet-23slim_SwissOkutama.pth',
                 'ddrnet39': 'DDRNet39_1080_SwissOkutama.pth',
                 'bisenet': 'BiSeNetv2_ResNet-18_SwissOkutama.pth'}
    
    dlist = []
    
    for item in hrnet_direc:
        dlist.append(os.path.join(hrnet_dir, item, 'best.pth'))
    
    for key in model_path.keys():
        dlist.append(os.path.join(model_path[key], model_dict[key]))
        
    # print(dlist)
    return dlist
    
if __name__ == '__main__':
    sv_pred = True
    
    # configure input image
    base_dir = '/home/rigi/segmentation/datasets/Okutama-Swiss-dataset/regions'
    sv_path = '/home/rigi/thesis/images/results'
    filelist = {
                'K': ['swiss_IMG_8725_0,1.png', 'swiss_IMG_8740_0,0.png', 'swiss_IMG_8793_0,1.png'],
                'D': ['okutama_04_90_013_1,1.png', 'okutama_04_90_014_1,1.png', 'okutama_04_90_016_1,1.png',
                'okutama_04_90_023_1,0.png'],
                # 'H': ['okutama_08_90_027_1,1.png'],
                # 'J': ['swiss_IMG_8775_0,1.png']
                }

    # for file in glob.glob(image_dir + "/images/*.png"):
    model_list = get_file_list()
    # print(len(model_list))

    
    for region in filelist.keys():
        image_dir = os.path.join(base_dir, region)
        print(image_dir)
        for file in filelist[region]:
            print(file)
            main(file)