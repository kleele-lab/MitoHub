import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
import glob
import gc
import cv2
from os.path import basename

# Load tiff stack images and masks

files_images = glob.glob('segment_dataset/images/*.png')
files_masks = glob.glob('segment_dataset/masks/*.png')

patch_size_w = 320
patch_size_h = 229
step = 229

all_img_patches = []

for x in range(len(files_images)):

    large_image = cv2.imread(files_images[x], cv2.IMREAD_GRAYSCALE)
    #large_image = cv2.resize(large_image, (1024, 768), interpolation = cv2.INTER_AREA)
    large_image = np.array(large_image)
    
    patches_img = patchify(large_image, (patch_size_w, patch_size_h), step=step)  #Step=256 for 256 patches means no overlap
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:]
            cv2.imwrite('patches/images/image_'+str(basename(files_images[x])).split('.')[0]+str(i)+str(j)+'.png', single_patch_img)
            print('patches/images/image_'+str(basename(files_images[x])).split('.')[0]+str(i)+str(j)+'.png')
            
            all_img_patches.append(single_patch_img)
            gc.collect()
        gc.collect()
    #print(np.array(all_img_patches).shape)
    gc.collect()

images = np.array(all_img_patches)
np.save('patchify_images.npy', images)

print(images.shape)

all_mask_patches = []

for x in range(len(files_masks)):
    large_mask = cv2.imread(files_masks[x] , cv2.IMREAD_GRAYSCALE)

    large_mask = np.array(large_mask)
    
    patches_mask = patchify(large_mask, (patch_size_w, patch_size_h), step=step)  #Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i,j,:,:]
            cv2.imwrite('patches/masks/image_'+str(basename(files_images[x])).split('.')[0]+str(i)+str(j)+'.png', single_patch_mask)
            print('patches/masks/image_'+str(basename(files_images[x])).split('.')[0]+str(i)+str(j)+'.png')
            
            single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
            all_mask_patches.append(single_patch_mask)
            gc.collect()
        gc.collect()
    #print(np.array(all_mask_patches).shape)
    gc.collect()

masks = np.array(all_mask_patches)
#np.save('patchify_masks.npy', masks)

print(masks.shape)

# Create a list to store the indices of non-empty masks
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
# Filter the image and mask arrays to keep only the non-empty pairs
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]

#np.save('patchify_images_filt.npy', filtered_images)
#np.save('patchify_masks_filt.npy', filtered_masks)