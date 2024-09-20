# Python In-built packages
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
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from patched_yolo_infer import CombineDetections#, MakeCropsDetectThem, visualize_results
import random
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import shutil
import gc

class MakeCropsDetectThem:

    def __init__(
        self,
        image: np.ndarray,
        model_path="yolov8m.pt",
        imgsz=640,
        conf=0.5,
        iou=0.7,
        classes_list=None,
        segment=False,
        shape_x=700,
        shape_y=600,
        overlap_x=25,
        overlap_y=25,
        show_crops=False,
        resize_initial_size=False,
        model=None,
        memory_optimize=True,
        inference_extra_args=None,
    ) -> None:
        if model is None:
            self.model = YOLO(model_path)  # Load the model from the specified path
        else:
            self.model = model
        self.image = image  # Input image
        self.imgsz = imgsz  # Size of the input image for inference
        self.conf = conf  # Confidence threshold for detections
        self.iou = iou  # IoU threshold for non-maximum suppression
        self.classes_list = classes_list  # Classes to detect
        self.segment = segment  # Whether to perform segmentation
        self.shape_x = shape_x  # Size of the crop in the x-coordinate
        self.shape_y = shape_y  # Size of the crop in the y-coordinate
        self.overlap_x = overlap_x  # Percentage of overlap along the x-axis
        self.overlap_y = overlap_y  # Percentage of overlap along the y-axis
        self.crops = []  # List to store the CropElement objects
        self.show_crops = show_crops  # Whether to visualize the cropping
        self.resize_initial_size = resize_initial_size  # slow operation !
        self.memory_optimize = memory_optimize # memory opimization option for segmentation
        self.class_names_dict = self.model.names # dict with human-readable class names
        self.inference_extra_args = inference_extra_args # dict with extra ultralytics inference parameters

        self.crops = self.get_crops_xy(
            self.image,
            shape_x=self.shape_x,
            shape_y=self.shape_y,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            show=self.show_crops,
        )
        self._detect_objects()

    def get_crops_xy(
        self,
        image_full,
        shape_x: int,
        shape_y: int,
        overlap_x=25,
        overlap_y=25,
        show=False,
    ):

        cross_koef_x = 1 - (overlap_x / 100)
        cross_koef_y = 1 - (overlap_y / 100)

        data_all_crops = []

        y_steps = int((image_full.shape[0] - shape_y) / (shape_y * cross_koef_y)) + 1
        x_steps = int((image_full.shape[1] - shape_x) / (shape_x * cross_koef_x)) + 1

        y_new = round((y_steps-1) * (shape_y * cross_koef_y) + shape_y)
        x_new = round((x_steps-1) * (shape_x * cross_koef_x) + shape_x)
        image_innitial = image_full.copy()
        image_full = cv2.resize(image_full, (x_new, y_new))

        if show:
            plt.figure(figsize=[x_steps*0.9, y_steps*0.9])

        count = 0
        for i in range(y_steps):
            for j in range(x_steps):
                x_start = int(shape_x * j * cross_koef_x)
                y_start = int(shape_y * i * cross_koef_y)

                # Check for residuals
                if x_start + shape_x > image_full.shape[1]:
                    print('Error in generating crops along the x-axis')
                    continue
                if y_start + shape_y > image_full.shape[0]:
                    print('Error in generating crops along the y-axis')
                    continue

                im_temp = image_full[y_start:y_start + shape_y, x_start:x_start + shape_x]

                # Display the result:
                if show:
                    plt.subplot(y_steps, x_steps, i * x_steps + j + 1)
                    plt.imshow(cv2.cvtColor(im_temp.copy(), cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                count += 1

                data_all_crops.append(CropElement(
                                        source_image=image_innitial,
                                        source_image_resized=image_full,
                                        crop=im_temp,
                                        number_of_crop=count,
                                        x_start=x_start,
                                        y_start=y_start,
                ))

        if show:
            #plt.show()
            plt.savefig(str(output_dir_path)+'_'+str(current_dateTime)+'/patches_'+str(basename(file)), 
                        dpi=500, bbox_inches='tight')
            print('Number of generated images:', count)
            st.write('Number of generated images: '+str(count))

        return data_all_crops

    def _detect_objects(self):
        """
        Method to detect objects in each crop.

        This method iterates through each crop, performs inference using the YOLO model,
        calculates real values, and optionally resizes the results.

        Returns:
            None
        """
        for crop in self.crops:
            crop.calculate_inference(
                self.model,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                segment=self.segment,
                classes_list=self.classes_list,
                memory_optimize=self.memory_optimize,
                extra_args=self.inference_extra_args
            )
            crop.calculate_real_values()
            if self.resize_initial_size:
                crop.resize_results()

import numpy as np
import cv2


class CropElement:
    # Class containing information about a specific crop
    def __init__(
        self,
        source_image: np.ndarray,
        source_image_resized: np.ndarray,
        crop: np.ndarray,
        number_of_crop: int,
        x_start: int,
        y_start: int
    ) -> None:
        self.source_image = source_image  # Original image 
        self.source_image_resized = source_image_resized  # Original image (resized to a multiple of the crop size)
        self.crop = crop  # Specific crop 
        self.number_of_crop = number_of_crop  # Crop number in order from left to right, top to bottom
        self.x_start = x_start  # Coordinate of the top-left corner X
        self.y_start = y_start  # Coordinate of the top-left corner Y

        # YOLO output results:
        self.detected_conf = None  # List of confidence scores of detected objects
        self.detected_cls = None  # List of classes of detected objects
        self.detected_xyxy = None  # List of lists containing xyxy box coordinates
        self.detected_masks = None # List of np arrays containing masks in case of yolo-seg
        self.polygons = None # List of polygons points in case of using memory optimaze
        
        # Refined coordinates according to crop position information
        self.detected_xyxy_real = None  # List of lists containing xyxy box coordinates in values from source_image_resized or source_image
        self.detected_masks_real = None # List of np arrays containing masks in case of yolo-seg with the size of source_image_resized or source_image
        self.detected_polygons_real = None # List of polygons points in case of using memory optimaze in values from source_image_resized or source_image

    def calculate_inference(self, model, imgsz=640, conf=0.35, iou=0.7, segment=False, classes_list=None, memory_optimize=False, extra_args=None):

        # Perform inference
        extra_args = {} if extra_args is None else extra_args
        predictions = model.predict(self.crop, imgsz=imgsz, conf=conf, iou=iou, classes=classes_list, verbose=False, **extra_args)

        pred = predictions[0]

        # Get the bounding boxes and convert them to a list of lists
        self.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()

        # Get the classes and convert them to a list
        self.detected_cls = pred.boxes.cls.cpu().int().tolist()

        # Get the mask confidence scores
        self.detected_conf = pred.boxes.conf.cpu().numpy()

        if segment and len(self.detected_cls) != 0:
            if memory_optimize:
                # Get the polygons
                self.polygons = [mask.astype(np.uint16) for mask in pred.masks.xy]
            else:
                # Get the masks
                self.detected_masks = pred.masks.data.cpu().numpy()
            

    def calculate_real_values(self):
        # Calculate real values of bboxes and masks in source_image_resized
        x_start_global = self.x_start  # Global X coordinate of the crop
        y_start_global = self.y_start  # Global Y coordinate of the crop

        self.detected_xyxy_real = []  # List of lists with xyxy box coordinates in the values ​​of the source_image_resized
        self.detected_masks_real = []  # List of np arrays with masks in case of yolo-seg sized as source_image_resized
        self.detected_polygons_real = [] # List of polygons in case of yolo-seg sized as source_image_resized

        for bbox in self.detected_xyxy:
            # Calculate real box coordinates based on the position information of the crop
            x_min, y_min, x_max, y_max = bbox
            x_min_real = x_min + x_start_global
            y_min_real = y_min + y_start_global
            x_max_real = x_max + x_start_global
            y_max_real = y_max + y_start_global
            self.detected_xyxy_real.append([x_min_real, y_min_real, x_max_real, y_max_real])

        if self.detected_masks is not None:
            for mask in self.detected_masks:
                # Create a black image with the same size as the source image
                black_image = np.zeros((self.source_image_resized.shape[0], self.source_image_resized.shape[1]))
                mask_resized = cv2.resize(np.array(mask).copy(), (self.crop.shape[1], self.crop.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # Place the mask in the correct position on the black image
                black_image[y_start_global:y_start_global+self.crop.shape[0],
                            x_start_global:x_start_global+self.crop.shape[1]] = mask_resized

                # Append the masked image to the list of detected_masks_real
                self.detected_masks_real.append(black_image)

        if self.polygons is not None:
            # Adjust the mask coordinates
            for mask in self.polygons:
                mask[:, 0] += x_start_global  # Add x_start_global to all x coordinates
                mask[:, 1] += y_start_global  # Add y_start_global to all y coordinates
                self.detected_polygons_real.append(mask.astype(np.uint16))
        
    def resize_results(self):
        # from source_image_resized to source_image sizes transformation
        resized_xyxy = []
        resized_masks = []
        resized_polygons = []

        for bbox in self.detected_xyxy_real:
            # Resize bbox coordinates
            x_min, y_min, x_max, y_max = bbox
            x_min_resized = int(x_min * (self.source_image.shape[1] / self.source_image_resized.shape[1]))
            y_min_resized = int(y_min * (self.source_image.shape[0] / self.source_image_resized.shape[0]))
            x_max_resized = int(x_max * (self.source_image.shape[1] / self.source_image_resized.shape[1]))
            y_max_resized = int(y_max * (self.source_image.shape[0] / self.source_image_resized.shape[0]))
            resized_xyxy.append([x_min_resized, y_min_resized, x_max_resized, y_max_resized])

        for mask in self.detected_masks_real:
            # Resize mask
            mask_resized = cv2.resize(mask, (self.source_image.shape[1], self.source_image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            resized_masks.append(mask_resized)


        for polygon in self.detected_polygons_real:
            polygon[:, 0] = (polygon[:, 0] * (self.source_image.shape[1] / self.source_image_resized.shape[1])).astype(np.uint16)
            polygon[:, 1] = (polygon[:, 1] * (self.source_image.shape[0] / self.source_image_resized.shape[0])).astype(np.uint16)
            resized_polygons.append(polygon)

        self.detected_xyxy_real = resized_xyxy
        self.detected_masks_real = resized_masks
        self.detected_polygons_real = resized_polygons

def visualize_results(
    img,
    boxes,
    classes_ids,
    confidences=[],
    classes_names=[], 
    masks=[],
    polygons=[],
    segment=False,
    show_boxes=True,
    show_class=True,
    fill_mask=False,
    alpha=0.3,
    color_class_background=(0, 0, 255),
    color_class_text=(255, 255, 255),
    thickness=4,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.5,
    delta_colors=0,
    dpi=150,
    random_object_colors=False,
    show_confidences=False,
    axis_off=True,
    show_classes_list=[],
    return_image_array=False
):

    # Create a copy of the input image
    labeled_image = img.copy()
    labeled_image_mask = img.copy()

    if random_object_colors:
        random.seed(int(delta_colors))

    # Process each prediction
    for i in range(len(classes_ids)):
        # Get the class for the current detection
        if len(classes_names)>0:
            class_name = str(classes_names[i])
        else:
            class_name = str(classes_ids[i])

        if show_classes_list and int(classes_ids[i]) not in show_classes_list:
            continue

        if random_object_colors:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            # Assign color according to class
            random.seed(int(classes_ids[i] + delta_colors))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        box = boxes[i]
        x_min, y_min, x_max, y_max = box

        if segment and len(masks) > 0:
            mask = masks[i]
            # Resize mask to the size of the original image using nearest neighbor interpolation
            mask_resized = cv2.resize(
                np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            # Add label to the mask
            mask_contours, _ = cv2.findContours(
                mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if fill_mask:
                if alpha == 1:
                    cv2.fillPoly(labeled_image, pts=mask_contours, color=color)
                else:
                    color_mask = np.zeros_like(img)
                    color_mask[mask_resized > 0] = color
                    labeled_image = cv2.addWeighted(labeled_image, 1, color_mask, alpha, 0)
            
            cv2.drawContours(labeled_image, mask_contours, -1, color, thickness)
            cv2.fillPoly(labeled_image_mask, pts=mask_contours, color=(255,255,255))
            labeled_image_mask = cv2.threshold(labeled_image_mask, 128, 255, cv2.THRESH_BINARY)[1]        
        
        elif segment and len(polygons) > 0:
            if len(polygons[i]) > 0:
                points = np.array(polygons[i].reshape((-1, 1, 2)), dtype=np.int32)
                if fill_mask:
                    if alpha == 1:
                        cv2.fillPoly(labeled_image, pts=[points], color=color)
                    else:
                        mask_from_poly = np.zeros_like(img)
                        color_mask_from_poly = cv2.fillPoly(mask_from_poly, pts=[points], color=color)
                        labeled_image = cv2.addWeighted(labeled_image, 1, color_mask_from_poly, alpha, 0)
                cv2.drawContours(labeled_image, [points], -1, color, thickness)

        # Write class label
        if show_boxes:
            cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

        if show_class:
            if show_confidences:
                label = f'{str(class_name)} {confidences[i]:.2}'
            else:
                label = str(class_name)
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(
                labeled_image,
                (x_min, y_min),
                (x_min + text_width + 5, y_min + text_height + 5),
                color_class_background,
                -1,
            )
            cv2.putText(
                labeled_image,
                label,
                (x_min + 5, y_min + text_height),
                font,
                font_scale,
                color_class_text,
                thickness=thickness,
            )

    if return_image_array:
        return labeled_image
    else:
        # Display the final image with overlaid masks and labels
        plt.figure(figsize=(8, 8), dpi=dpi)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        plt.imshow(labeled_image)
        if axis_off:
            plt.axis('off')
        #plt.show()
        plt.savefig(str(output_dir_path)+'_'+str(current_dateTime)+'/segment_mask_'+str(basename(file)), 
                    dpi=500, bbox_inches='tight', pad_inches = 0)
        
        plt.imshow(labeled_image_mask)
        plt.savefig(str(output_dir_path)+'_'+str(current_dateTime)+'/mask_'+str(basename(file)), 
                    dpi=500, bbox_inches='tight', pad_inches = 0)

        tifffile.imwrite(str(output_dir_path)+'_'+str(current_dateTime)+'/mask_'+str(basename(file)).split('.')[0]+'.tif', 
                         labeled_image_mask)

import glob

# Download YOLOv8 model
#yolov8_model_path = "segmentation_models/seg_yolov8x_patches_200e_640_last.pt"
yolov8_model_path = "MitoHub_models/segmentation_models/segmentation_model_yolov8x_patches_500_640_last.pt"

model_names = glob.glob('MitoHub_models/segmentation_models/*.pt')

# arrange an instance segmentation model for test
st.set_page_config(page_title="Segmentation Batch Processing", page_icon="📈")
st.sidebar.header("Segmentation Batch Processing")

st.title("Mitochondrial Segmentation Batch Processing")

input_folder = st.text_input("Please enter the path to the Folder containing images to be segmented", "")
trained_model = st.selectbox("Select the trained model to use for segmentation",
                             model_names)

yolov8_model_path = trained_model

confidence_threshold = st.number_input("Please type the confidence threshold you'd like to use ", value=0.1)
patch_width = st.number_input("Please type the width of the patch to use ", value=340)
patch_height = st.number_input("Please type the height of the patch to use ", value=299)
overlap_input = st.number_input("Please type the Patch overlap that you would like to set ", value=1)
output_dir_path = st.text_input("Output Directory Path")

if st.button("Process Batch"):
    new_dateTime = datetime.now()
    current_dateTime = new_dateTime.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(str(output_dir_path)+'_'+str(current_dateTime))
    
    files = glob.glob(os.path.join(input_folder,'*'))
    for file in files:
        st.write('Processing File '+str(file))
        # Convert the file to an opencv image.
        img_path = str(file)
        img = cv2.imread(img_path)

        image = Image.open(file)
        img_array = np.array(image)

        element_crops = MakeCropsDetectThem(
            image=img,
            model_path=yolov8_model_path,
            segment=True,
            show_crops=True,
            shape_x=patch_width,
            shape_y=patch_height,
            overlap_x=int(overlap_input),
            overlap_y=int(overlap_input),
            conf=float(confidence_threshold),
            classes_list=[0],
            resize_initial_size=True,
            memory_optimize=False
        )

        result = CombineDetections(element_crops, match_metric='IOS')
        crop_file = open(str(output_dir_path)+'_'+str(current_dateTime)+'/patches_'+str(basename(file)),'rb')
        # Convert the file to an opencv image.
        crop_image_out = Image.open(crop_file)
        crop_img_array_out = np.array(crop_image_out)
        # Now do something with the image! For example, let's display it:
        
        # Final Results (can be saved to get desired results):
        img_out=result.image
        confidences=result.filtered_confidences
        boxes=result.filtered_boxes
        #polygons=result.filtered_polygons
        classes_ids=result.filtered_classes_id
        classes_names=result.filtered_classes_names
        masks = result.filtered_masks

        # Visualizing the results using the visualize_results function
        visualize_results(
            img=result.image,
            show_boxes=False,
            show_class=False,
            confidences=result.filtered_confidences,
            boxes=result.filtered_boxes,
            #polygons=result.filtered_polygons,
            masks=result.filtered_masks,
            classes_ids=result.filtered_classes_id,
            classes_names=result.filtered_classes_names,
            segment=True,
        )
        
        output_file = open(str(output_dir_path)+'_'+str(current_dateTime)+'/segment_mask_'+str(basename(file)),'rb')
        # Convert the file to an opencv image.
        image_out = Image.open(output_file)
        img_array_out = np.array(image_out)
        gc.collect()
    gc.collect()
    st.write("Files Processed and saved at :"+str(output_dir_path))