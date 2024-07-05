import numpy as np
import io
import os
import glob
import gc
from os.path import basename
import h5py
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from skimage import io

files = glob.glob('Segmentation/*.h5')

for file in files:

    f_mask = h5py.File(str(file), 'r')

    f_mask_np = np.matrix(f_mask['exported_data'][:,:])

    f_mask_np[f_mask_np == 2] = 0

    print(f_mask_np)

    plt.margins(x=0)
    plt.axis('off')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.imshow(f_mask_np, cmap='gray')
    plt.savefig('mask_set/'+str(basename(file)).split('_decon')[0]+'_decon.png', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.cla()
    plt.clf()
    gc.collect()
