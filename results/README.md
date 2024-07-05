## Results

### Mobility Estimation
We carry out Mobility Estimation using Shi-Tomasi Corner detection applied to every alternative frame in order to find several corner points that are tracked using the Lukas Kanade Optical flow estimation method. The workflow for the same is as follows:

![alt text](https://github.com/kleele-lab/MitoHub/blob/main/results/mobility/workflow.png)

### Segmentation Model
Our segmentation model was able to achieve a **mAP50 score of 93%** and a **mAP50-95 score of 70%**. We trained the model for 500 epochs by dividing each image and segmentation masks into patches and re-training. 

The segmentation model was trained using the following pipeline:

![alt text](https://github.com/kleele-lab/MitoHub/blob/main/results/segmentation/workflow.png)

We have attached the results as follows at the end of 500 epochs and 1100 epochs:

#### 500 epochs, 16 batch size
![alt text](https://github.com/kleele-lab/MitoHub/blob/main/results/segmentation/patches_yolov8x_seg_500e/results.png)

#### 1100 epochs, 16 batch size
![alt text](https://github.com/kleele-lab/MitoHub/blob/main/results/segmentation/patches_yolov8x_seg_1100e/results.png)


When we did not use patches based method, we were only able to achieve a **mAP50 score of 55%** and a **mAP50-95 score of 25%** over an image size of 1280 and 300 epochs.

![alt text](https://github.com/kleele-lab/MitoHub/blob/main/results/segmentation/yolov8x_300e_1280/results.png)
