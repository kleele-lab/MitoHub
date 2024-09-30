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

files = glob.glob('Raw_data/*.h5')

for file in files:

    f = h5py.File(str(file), 'r')

    f_np = np.matrix(f['data'])

    print(f_np)

    plt.margins(x=0)
    plt.axis('off')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.imshow(f_np, cmap='gray')
    plt.savefig('image_set/'+str(basename(file)).split('.')[0]+'.png', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.cla()
    plt.clf()
    gc.collect()
