import cv2
import os
import argparse
import csv
from tqdm import tqdm
import albumentations as A

def main(args):
    with open(args.regions, 'r') as f:
        next(f)  # skip header
        reader = csv.reader(f, delimiter=',')
        data = [[r[0], int(r[1]), int(r[2]), r[3]] for r in reader]

    divX = 2
    divY = 2
    
    ratio = 4

    transform = get_transform(1920, 1080) #w, h

    for imgFile, x, y, region in tqdm(data):
        exportCell(args, imgFile, divX, divY, x, y, region, transform, ratio)


# Possible image transformations during augmentation
def get_transform(width, height):
    transform = A.Compose([
        A.Rotate(limit=[-180, 180], p=0.3, interpolation=cv2.INTER_NEAREST),
        A.RandomCrop(width=width, height=height, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=0.15, p=0.2),            
    ], p=1.0)

    return transform


def augment_dataset(img, mask, transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #input in BGR
    
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed['image']
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    transformed_mask = transformed['mask']
    
    return transformed_image, transformed_mask
    
    
def exportCell(args, imgName, divX, divY, x, y, region, transform, ratio):
    if region == 'unused':
        return

    imgFile = os.path.join(args.datasetdir, 'images', imgName)
    img = cv2.imread(imgFile)
    if img is None:
        raise Exception(f'Could not read {imgFile}')
    imgH, imgW = img.shape[0:2]

    name = os.path.splitext(imgName)[0]

    gtFile = os.path.join(args.datasetdir, 'ground_truth', name + '.png')

    gtImg = cv2.imread(gtFile)
    if gtImg is None:
        raise Exception(f'Could not read {gtFile}')

    cellW, cellH = int(imgW / divX), int(imgH / divY)
    cellX = int(x * cellW)
    cellY = int(y * cellH)

    cell = img[cellY:cellY+cellH, cellX:cellX+cellW]
    gtCell = gtImg[cellY:cellY+cellH, cellX:cellX+cellW]

    # remove {train|test|val} prefix
    name = os.path.split(name)[1]

    cellFilename = f'{name}_{x},{y}.png'
    cellPath = os.path.join(
        args.outdir, region, 'images', cellFilename)
    gtCellPath = os.path.join(
        args.outdir, region, 'ground_truth', cellFilename)
        
    os.makedirs(os.path.split(cellPath)[0], exist_ok=True)
    os.makedirs(os.path.split(gtCellPath)[0], exist_ok=True)
    

    cellPath = os.path.join(
        args.outdir, region, 'images')
    gtCellPath = os.path.join(
        args.outdir, region, 'ground_truth')
    
    for i in range(ratio):
        transformed_image, transformed_mask = augment_dataset(cell, gtCell, transform)
        filename_aug = f'{name}_{x},{y}_{i+1}.png'
        
        cv2.imwrite(os.path.join(cellPath, filename_aug), transformed_image)
        cv2.imwrite(os.path.join(gtCellPath, filename_aug), transformed_mask)
                
                

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Generate Okutama and Swiss datasets regions')
    ap.add_argument('--regions', type=str,
                    help='File with list of images, in CSV',
                    default='regions.csv')
    ap.add_argument('--datasetdir', type=str,
                    help='Directory of Okutama-Swiss dataset, with {images|ground_truth}/{train|val|test} subdirs',
                    default='.')
    ap.add_argument('--outdir', type=str,
                    help='Directory to save regions. 12 subdirs will be created',
                    default='regions_aug/')

    args = ap.parse_args()
    main(args)
