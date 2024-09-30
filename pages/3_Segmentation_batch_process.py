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
import torch

class MakeCropsDetectThem:

    def __init__(
        self,
        image: np.ndarray,
        model_path="yolov8m.pt",
        imgsz=640,
        conf=0.35,
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


class CombineDetections:
    """
    Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression).

    Args:
        element_crops (MakeCropsDetectThem): Object containing crop information.
        nms_threshold (float): IoU/IoS threshold for non-maximum suppression.  Dafault is 0.3.
        match_metric (str): Matching metric, either 'IOU' or 'IOS'. Dafault is IoS.
        class_agnostic_nms (bool) Determines the NMS mode in object detection. When set to True, NMS 
            operates across all classes, ignoring class distinctions and suppressing less confident 
            bounding boxes globally. Otherwise, NMS is applied separately for each class. Default is True.
        intelligent_sorter (bool): Enable sorting by area and rounded confidence parameter. 
            If False, sorting will be done only by confidence (usual nms). Dafault is True.
        sorter_bins (int): Number of bins to use for intelligent_sorter. A smaller number of bins makes
            the NMS more reliant on object sizes rather than confidence scores. Defaults to 5.

    Attributes:
        class_names (dict): Dictionary containing class names of yolo model.
        crops (list): List to store the CropElement objects.
        image (np.ndarray): Source image in BGR.
        nms_threshold (float): IOU/IOS threshold for non-maximum suppression.
        match_metric (str): Matching metric (IOU/IOS).
        class_agnostic_nms (bool) Determines the NMS mode in object detection.
        intelligent_sorter (bool): Flag indicating whether sorting by area and confidence parameter is enabled.
        sorter_bins (int): Number of bins to use for intelligent_sorter. 
        detected_conf_list_full (list): List of detected confidences.
        detected_xyxy_list_full (list): List of detected bounding boxes.
        detected_masks_list_full (list): List of detected masks.
        detected_polygons_list_full (list): List of detected polygons when memory optimization is enabled.
        detected_cls_id_list_full (list): List of detected class IDs.
        detected_cls_names_list_full (list): List of detected class names.
        filtered_indices (list): List of indices after non-maximum suppression.
        filtered_confidences (list): List of confidences after non-maximum suppression.
        filtered_boxes (list): List of bounding boxes after non-maximum suppression.
        filtered_classes_id (list): List of class IDs after non-maximum suppression.
        filtered_classes_names (list): List of class names after non-maximum suppression.
        filtered_masks (list): List of filtered (after nms) masks if segmentation is enabled.
        filtered_polygons (list): List of filtered (after nms) polygons if segmentation and
            memory optimization are enabled.
    """

    def __init__(
        self,
        element_crops: MakeCropsDetectThem,
        nms_threshold=0.3,
        match_metric='IOS',
        intelligent_sorter=True,
        sorter_bins=5,
        class_agnostic_nms=True
    ) -> None:
        self.class_names = element_crops.class_names_dict 
        self.crops = element_crops.crops  # List to store the CropElement objects
        if element_crops.resize_initial_size:
            self.image = element_crops.crops[0].source_image
        else:
            self.image = element_crops.crops[0].source_image_resized

        self.nms_threshold = nms_threshold  # IOU or IOS treshold for NMS
        self.match_metric = match_metric 
        self.intelligent_sorter = intelligent_sorter # enable sorting by area and confidence parameter
        self.sorter_bins = sorter_bins
        self.class_agnostic_nms = class_agnostic_nms

        # Combinate detections of all patches
        (
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.detected_masks_list_full,
            self.detected_cls_id_list_full,
            self.detected_polygons_list_full
        ) = self.combinate_detections(crops=self.crops)

        self.detected_cls_names_list_full = [
            self.class_names[value] for value in self.detected_cls_id_list_full
        ]  # make str list

        # Invoke the NMS:
        if self.class_agnostic_nms:
            self.filtered_indices = self.nms(
                torch.tensor(self.detected_conf_list_full),
                torch.tensor(self.detected_xyxy_list_full),
                self.match_metric,
                self.nms_threshold,
                self.detected_masks_list_full,
                intelligent_sorter=self.intelligent_sorter
            ) 

        else:
            self.filtered_indices = self.not_agnostic_nms(
                torch.tensor(self.detected_cls_id_list_full),
                torch.tensor(self.detected_conf_list_full),
                torch.tensor(self.detected_xyxy_list_full),
                self.match_metric,
                self.nms_threshold,
                self.detected_masks_list_full,
                intelligent_sorter=self.intelligent_sorter
            )  

        # Apply filtering (nms output indeces) to the prediction lists
        self.filtered_confidences = [self.detected_conf_list_full[i] for i in self.filtered_indices]
        self.filtered_boxes = [self.detected_xyxy_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_id = [self.detected_cls_id_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_names = [self.detected_cls_names_list_full[i] for i in self.filtered_indices]

        # Masks filtering:
        if element_crops.segment and not element_crops.memory_optimize:
            self.filtered_masks = [self.detected_masks_list_full[i] for i in self.filtered_indices]
        else:
            self.filtered_masks = []

        # Polygons filtering:
        if element_crops.segment and element_crops.memory_optimize:
            self.filtered_polygons = [self.detected_polygons_list_full[i] for i in self.filtered_indices]
        else:
            self.filtered_polygons = []

    def combinate_detections(self, crops):
        """
        Combine detections from multiple crop elements.

        Args:
            crops (list): List of CropElement objects.

        Returns:
            tuple: Tuple containing lists of detected confidences, bounding boxes,
                masks, and class IDs.
        """
        detected_conf = []
        detected_xyxy = []
        detected_masks = []
        detected_cls = []
        detected_polygons = []

        for crop in crops:
            detected_conf.extend(crop.detected_conf)
            detected_xyxy.extend(crop.detected_xyxy_real)
            detected_masks.extend(crop.detected_masks_real)
            detected_cls.extend(crop.detected_cls)
            detected_polygons.extend(crop.detected_polygons_real)

        return detected_conf, detected_xyxy, detected_masks, detected_cls, detected_polygons

    @staticmethod
    def average_to_bound(confidences, N=10):
        """
        Bins the given confidences into N equal intervals between 0 and 1, 
        and rounds each confidence to the left boundary of the corresponding bin.

        Parameters:
        confidences (list or np.array): List of confidence values to be binned.
        N (int, optional): Number of bins to use. Defaults to 10.

        Returns:
        list: List of rounded confidence values, each bound to the left boundary of its bin.
        """
        # Create the bounds
        step = 1 / N
        bounds = np.arange(0, 1 + step, step)

        # Use np.digitize to determine the corresponding bin for each value
        indices = np.digitize(confidences, bounds, right=True) - 1

        # Bind values to the left boundary of the corresponding bin
        averaged_confidences = np.round(bounds[indices], 2) 

        return averaged_confidences.tolist()

    @staticmethod
    def intersect_over_union(mask, masks_list):
        """
        Compute Intersection over Union (IoU) scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask.
        """
        iou_scores = []
        for other_mask in masks_list:
            # Compute intersection and union
            intersection = np.logical_and(mask, other_mask).sum()
            union = np.logical_or(mask, other_mask).sum()
            # Compute IoU score, avoiding division by zero
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)
        return torch.tensor(iou_scores)

    @staticmethod
    def intersect_over_smaller(mask, masks_list):
        """
        Compute Intersection over Smaller area scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask,
                calculated over the smaller area.
        """
        ios_scores = []
        for other_mask in masks_list:
            # Compute intersection and area of smaller mask
            intersection = np.logical_and(mask, other_mask).sum()
            smaller_area = min(mask.sum(), other_mask.sum())
            # Compute IoU score over smaller area, avoiding division by zero
            ios = intersection / smaller_area if smaller_area != 0 else 0
            ios_scores.append(ios)
        return torch.tensor(ios_scores)

    def nms(
        self,
        confidences: torch.tensor,
        boxes: torch.tensor,
        match_metric,
        nms_threshold,
        masks=[],
        intelligent_sorter=False, 
        cls_indexes=None 
    ):
        """
        Apply class-agnostic non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.

        Args:
            confidences (torch.Tensor): List of confidence scores.
            boxes (torch.Tensor): List of bounding boxes.
            match_metric (str): Matching metric, either 'IOU' or 'IOS'.
            nms_threshold (float): The threshold for match metric.
            masks (list): List of masks. 
            intelligent_sorter (bool, optional): intelligent sorter 
            cls_indexes (torch.Tensor):  indexes from network detections corresponding
                to the defined class,  uses in case of not agnostic nms

        Returns:
            list: List of filtered indexes.
        """
        if len(boxes) == 0:
            return []

        # Extract coordinates for every prediction box present
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of every box
        areas = (x2 - x1) * (y2 - y1)

        # Sort the prediction boxes according to their confidence scores or intelligent_sorter mode
        if intelligent_sorter:
            # Sort the prediction boxes according to their round confidence scores and area sizes
            order = torch.tensor(
                sorted(
                    range(len(confidences)),
                    key=lambda k: (
                        self.average_to_bound(confidences[k].item(), self.sorter_bins),
                        areas[k],
                    ),
                    reverse=False,
                )
            )
        else:
            order = confidences.argsort()
        # Initialise an empty list for filtered prediction boxes
        keep = []

        while len(order) > 0:
            # Extract the index of the prediction with highest score
            idx = order[-1]

            # Push the index in filtered predictions list
            keep.append(idx.tolist())

            # Remove the index from the list
            order = order[:-1]

            # If there are no more boxes, break
            if len(order) == 0:
                break

            # Select coordinates of BBoxes according to the indices
            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)

            # Find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            # Find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1

            # Take max with 0.0 to avoid negative width and height
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            # Find the intersection area
            inter = w * h

            # Find the areas of BBoxes
            rem_areas = torch.index_select(areas, dim=0, index=order)

            if match_metric == "IOU":
                # Find the union of every prediction with the prediction
                union = (rem_areas - inter) + areas[idx]
                # Find the IoU of every prediction
                match_metric_value = inter / union

            elif match_metric == "IOS":
                # Find the smaller area of every prediction with the prediction
                smaller = torch.min(rem_areas, areas[idx])
                # Find the IoU of every prediction
                match_metric_value = inter / smaller

            else:
                raise ValueError("Unknown matching metric")

            # If masks are provided and IoU based on bounding boxes is greater than 0,
            # calculate IoU for masks and keep the ones with IoU < nms_threshold
            if len(masks) > 0 and torch.any(match_metric_value > 0):

                mask_mask = match_metric_value > 0 

                order_2 = order[mask_mask]
                filtered_masks = [masks[i] for i in order_2]

                if match_metric == "IOU":
                    mask_iou = self.intersect_over_union(masks[idx], filtered_masks)
                    mask_mask = mask_iou > nms_threshold

                elif match_metric == "IOS":
                    mask_ios = self.intersect_over_smaller(masks[idx], filtered_masks)
                    mask_mask = mask_ios > nms_threshold
                # create a tensor of indences to delete in tensor order
                order_2 = order_2[mask_mask]
                inverse_mask = ~torch.isin(order, order_2)

                # Keep only those order values that are not contained in order_2
                order = order[inverse_mask]

            else:
                # Keep the boxes with IoU/IoS less than threshold
                mask = match_metric_value < nms_threshold

                order = order[mask]
        if cls_indexes is not None:
            keep = [cls_indexes[i] for i in keep]
        return keep

    def not_agnostic_nms(
            self,
            detected_cls_id_list_full,
            detected_conf_list_full, 
            detected_xyxy_list_full, 
            match_metric, 
            nms_threshold, 
            detected_masks_list_full, 
            intelligent_sorter
                     ):
        '''
            Performs Non-Maximum Suppression (NMS) in a non-agnostic manner, where NMS 
            is applied separately to each class.

            Args:
                detected_cls_id_list_full (torch.Tensor): tensor containing the class IDs for each detected object.
                detected_conf_list_full (torch.Tensor):  tensor of confidence scores.
                detected_xyxy_list_full (torch.Tensor): tensor of bounding boxes.
                match_metric (str): Matching metric, either 'IOU' or 'IOS'.
                nms_threshold (float): the threshold for match metric.
                detected_masks_list_full (torch.Tensor):  List of masks. 
                intelligent_sorter (bool, optional): intelligent sorter 

            Returns:
                List[int]: A list of indices representing the detections that are kept after applying
                    NMS for each class separately.

            Notes:
                - This method performs NMS separately for each class, which helps in
                    reducing false positives within each class.
                - If in your scenario, an object of one class can physically be inside
                    an object of another class, you should definitely use this non-agnostic nms
            '''
        all_keeps = []
        for cls in torch.unique(detected_cls_id_list_full):
            cls_indexes = torch.where(detected_cls_id_list_full==cls)[0]
            if len(detected_masks_list_full) > 0:
                masks_of_class = [detected_masks_list_full[i] for i in cls_indexes]
            else:
                masks_of_class = []
            keep_indexes = self.nms(
                    detected_conf_list_full[cls_indexes],
                    detected_xyxy_list_full[cls_indexes],
                    match_metric,
                    nms_threshold,
                    masks_of_class,
                    intelligent_sorter,
                    cls_indexes
                )
            all_keeps.extend(keep_indexes)
        return all_keeps



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
    #hi, wi, ci = img.shape

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
        labeled_image = cv2.resize(labeled_image, (wi, hi))
        labeled_image_mask = cv2.resize(labeled_image_mask, (wi, hi))
        plt.imshow(labeled_image)
        if axis_off:
            plt.axis('off')
        #plt.show()
        plt.savefig(str(output_dir_path)+'_'+str(current_dateTime)+'/segment_mask_'+str(basename(file)), 
                    dpi=500, bbox_inches='tight', pad_inches = 0)
        
        plt.imshow(labeled_image_mask)
        plt.savefig(str(output_dir_path)+'_'+str(current_dateTime)+'/mask_'+str(basename(file)), 
                    dpi=500, bbox_inches='tight', pad_inches = 0)
        labeled_image = cv2.resize(labeled_image, img.shape) 
        tifffile.imwrite(str(output_dir_path)+'_'+str(current_dateTime)+'/mask_'+str(basename(file)).split('.')[0]+'.tif', 
                         labeled_image_mask)

import glob
import gc

def tiff_proc(upload_tiff_file):
    img_path = upload_tiff_file
    image = tifffile.imread(img_path)
    print(image.shape)
    # let `video` be an array with dimensionality (T, H, W, C)
    image_n = image[:,:]
    imgplot = plt.imshow(image_n, cmap = 'gray')
    # Selecting the axis-X making the bottom and top axes False. 
    plt.tick_params(axis='x', which='both', bottom=False, 
                    top=False, labelbottom=False)         
    # Selecting the axis-Y making the right and left axes False 
    plt.tick_params(axis='y', which='both', right=False, 
                    left=False, labelleft=False) 
    plt.axis('off')
    plt.savefig(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(img_path)).split('.')[0]+'_input.png'), bbox_inches='tight', transparent="True", dpi=435, pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()
    gc.collect()

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
    log_file = open(str(output_dir_path)+'_'+str(current_dateTime)+"/logs.csv", "w+")
    log_file.write("Filename, trained_model_name, confidence_threshold, path_height, patch_width, patch_overlap\n")    
    files = glob.glob(os.path.join(input_folder,'*'))
    for file in files:
        st.write('Processing File '+str(file))
        # Convert the file to an opencv image.
        if str(file).split('.')[-1]=="tif":
            img_path = str(file)
            testimage = tifffile.imread(img_path)
            hi, wi = testimage.shape
            tiff_proc(img_path)
            img = cv2.imread(os.path.join(str(output_dir_path)+'_'+str(current_dateTime),str(basename(img_path)).split('.')[0]+'_input.png'))
        else:
            img_path = str(file)
            img = cv2.imread(img_path)
            hi, wi, ci = img.shape
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

        result = CombineDetections(element_crops, match_metric='IOU', intelligent_sorter=True, sorter_bins=5)
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
        logs_write = str(output_dir_path)+'_'+str(current_dateTime)+'/'+str(basename(file))+', '+str(trained_model)+', '+str(confidence_threshold)+', '+str(patch_height)+', '+str(patch_width)+', '+str(overlap_input)+'\n'
        log_file.write(logs_write)
        log_file.close()

        image_out = Image.open(output_file)
        img_array_out = np.array(image_out)
        gc.collect()
    gc.collect()
    st.write("Files Processed and saved!")