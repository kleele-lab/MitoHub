# Python In-built packages
from pathlib import Path
import PIL
from ultralytics import YOLO
# External packages
import streamlit as st
import io
import os
from os.path import basename
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import math
from datetime import datetime
import pandas as pd
import math
import tifffile as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gc
import glob

def lk_from_image(image):
    global idx, gray1, mask, prev
    if idx==0:
        gray1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        feature_params = dict(maxCorners = max_corner_value, qualityLevel = 0.0025, minDistance = 5, blockSize = 7)
        prev = cv2.goodFeaturesToTrack(gray1, mask = None, **feature_params)
        mask = np.zeros_like(image)
        idx+=1
        return image
    # Else
    im2 = image
    if idx%15==0:
        mask = np.zeros_like(image)
    
    if idx%2==0:
        # Every 2 images, relaunch the feature detection
        feature_params = dict(maxCorners = max_corner_value, qualityLevel = 0.0025, minDistance = 5, blockSize = 7)
        prev = cv2.goodFeaturesToTrack(gray1, mask = None, **feature_params)
    
    color = (255, 255, 0)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
 
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    next, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, prev, None, **lk_params)

    good_matches_old = prev[status == 1]
    good_matches_new = next[status == 1]

    for new, old in zip(good_matches_new, good_matches_old):
        a, b = new.ravel()
        c, d = old.ravel()
        r = math.sqrt((c-a)*(c-a)+(d-b)*(d-b))
        theta = math.degrees(math.atan(d-b/c-a))
        theta = (theta + 360) % 360
        
        #print(str(a)+', '+str(b)+', '+str(c)+', '+str(d)+', '+str(r))
        mask = cv2.arrowedLine(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
        #im2 = cv2.circle(im2, (int(a), int(b)), 3, color, -1)
        f.write(str(a)+', '+str(b)+', '+str(c)+', '+str(d)+', '+str(r)+', '+str(theta)+', ')
    f.write('\n')
    output = cv2.add(im2, mask)
    gray1 = gray2.copy()
    prev = good_matches_new.reshape(-1, 1, 2)
    idx += 1

    return output


def tiff_proc(upload_tiff_file):
    img_path = upload_tiff_file
    image = tf.imread(img_path)
    print(image.shape)
    # let `video` be an array with dimensionality (T, H, W, C)
    num_frames, height, width = image.shape
    os.mkdir(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_frames'))
    for i in range(0,num_frames):
        print("Writing Frame "+str(i))
        image_n = image[i,:,:]
        imgplot = plt.imshow(image_n, cmap = 'gray')
        # Selecting the axis-X making the bottom and top axes False. 
        plt.tick_params(axis='x', which='both', bottom=False, 
                        top=False, labelbottom=False)         
        # Selecting the axis-Y making the right and left axes False 
        plt.tick_params(axis='y', which='both', right=False, 
                        left=False, labelleft=False) 
        plt.axis('off')
        plt.savefig(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_frames/')+str(i)+'.png', bbox_inches='tight', transparent="True", dpi=435, pad_inches=0)
        plt.close()
        plt.cla()
        plt.clf()
        gc.collect()
    os.system("ffmpeg -framerate 20 -i "+os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_frames/')+"%d.png -vcodec libx264 -r 30 "+str(output_dir_path)+'_'+str(current_dateTime)+"/"+str(basename(file)).split('.')[0]+"_merged_video.mp4")


st.set_page_config(page_title="Mobility Batch Processing", page_icon="📈")
st.sidebar.header("Mobility Batch Processing")

st.title("Mobility Batch Processing")

uploaded_folder = st.text_input("Enter Input Folder Path", "")
max_corner_value = st.number_input("Enter the number of corner points to detect", value=0)
#input_labels = st.text_input("Please enter the path to input labels","")
output_dir_path = st.text_input("Enter Output Directory Path", "")

if st.button("Process Folder"):
    new_dateTime = datetime.now()
    current_dateTime = new_dateTime.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(str(output_dir_path)+'_'+str(current_dateTime))
    #if uploaded_file is not None:
    files = glob.glob(os.path.join(str(uploaded_folder),'*'))
    for file in files:
        if str(file).split('.')[-1] == 'tif' :
            idx = 0
            tiff_proc(file)
            video_file = os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_merged_video.mp4')
            
        else:
            idx = 0
            video_file = file

        f = open(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_sparse_csv.csv'), "w+")
        clip = VideoFileClip(video_file)

        white_clip = clip.fl_image(lk_from_image)

        #white_clip.write_videofile("test_lk_output.mp4",audio=False)
        st.write("Running Sparse Optical flow based mobility estimation")
        white_clip.write_videofile(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_sparse_flow.mp4'), codec="libx264")
        f.close() 

        df = pd.read_csv(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_sparse_csv.csv'), sep = ',', header=None)
        df_r = df.iloc[:,4]
        j = 0
        for i in range(1, int(len(df)/5)):
            df_r = pd.concat([df_r,df.iloc[:,i*5+5+j]], axis=1)
            j = j+1
        df_r = df_r.iloc[1:]
        df_r['Mean Mobility'] = df_r.mean(axis=1)
        print(df_r.head())
        df_r.to_csv(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(file)).split('.')[0]+'_mean_mobility.csv'),index = False)

    st.write('Process Finished, Results are stored at: '+str(output_dir_path))