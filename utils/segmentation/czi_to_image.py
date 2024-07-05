import czifile
import glob
import gc
from os.path import basename
import cv2

files = glob.glob('Day36/*')

for file in files:
    # Get an AICSImage object
    img = czifile.imread(str(file))
    #print(img.data)  # returns 6D STCZYX numpy array
    #print(img.dims)  # returns string "STCZYX"
    print(len(img[0][0][0]))

    for i in range(0, len(img[0][0][0])):
        print('output\\'+str(basename(file)).split('.')[0]+'\\frame_'+str(i)+'.png')
        #print(img[0][0][0][i][:][:][0])
        cv2.imwrite('output_36\\'+str(basename(file)).split('.')[0]+'_frame_'+str(i)+'.png', img[0][0][0][i][:][:][0])
        gc.collect()
    gc.collect()
