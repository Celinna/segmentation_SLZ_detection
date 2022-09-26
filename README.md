# Safe Landing Zone Selection with Segmentation Models

This repository contains the PyTorch based code used for my master's thesis titled "Semantic segmentation based emergency landing pipeline for UAVs". 
The dataset used for training is the [Swiss and Okutama Drone Datasets](https://www.okutama-segmentation.org/). 

## Contents
```
segmentation_SLZ_detection
│   README.md
│   test.py
│   test_real_world.py
│
└───configs
│   │   bisenet.yaml
│   │   ddrnet.yaml
│   │   ddrnet39.yaml
│   │   seg_hrnet_w32_config.yaml
│   │   seg_hrnet_w48_config.yaml
│
└───models
│   │   bisenetv2.py
│   │   ddrnet.py
│   │   ddrnet39.py
│   │   seg_hrnet.py
│   │   seg_hrnet_ocr.py
│
└───preprocessing
│   │   getFileLists.py
│   │   makeAugmentedData.py
│
└───training
│   │   HRNet-Semantic-Segmentation
│   │   semantic-segmentation
│   
└───utils
    │   coord_trans.py
    │   seg_model.py
    │   swiss_okutama.py
    │   utils.py
```

## Preprocessing
In addition to the preprocessing proposed by the original authors of the Swiss and Okutama Drone Datasets, 
two additional preprocessing files are provided.
1. Create region subsets by following the instructions from the Swiss and Okutama Drone Datasets.
2. Copy the files within the `preprocessing` folder to the dataset folder.
3. Use `makeAugmenteddata.py` to create augmented images for each dataset regions. The will be saved in a new folder titled `regions_aug`.
4. Use `getFileLists.py` to get a list of train/test/val file paths for each cross-validation fold. 

## Model Training and Testing
Training and testing was done using two separate repos. 
The high-performance network (High-Resolution Net) is trained using the HRNet-Semantic-Segmentation respository. 
The real-time networks (BiSeNetV2, DDRNet) are trained using the semantic-segmentation respository. 
Please see each individual READMEs for training process.

## Inference
Create a folder titled `output/` and place the trained models here. Some pre-trained models can be found [here](https://drive.google.com/file/d/1ACKSEFxI0ZktuV6y5Pe55caf0fxLaVQp/view?usp=sharing). In each test file, change the file-paths and model names accordingly.

Two testing files are available: 
* `test.py` performs inference on Swiss and Okutama dataset images.
* `test_real_world.py` performs inference on images taken from the IMX477 camera. 
