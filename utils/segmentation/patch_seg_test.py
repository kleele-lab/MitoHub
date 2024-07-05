from pathlib import Path

import cv2
from ultralytics import YOLO
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections#, visualize_results

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


img_path = 'segment_test.png'
img = cv2.imread(img_path)

# Download YOLOv8 model
yolov8_model_path = "segmentation_models/seg_yolov8x_patches_200e_640_last.pt"


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
    """
    Visualizes custom results of object detection or segmentation on an image.

    Args:
        img (numpy.ndarray): The input image in BGR format.
        boxes (list): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        classes_ids (list): A list of class IDs for each detection.
        confidences (list): A list of confidence scores corresponding to each bounding box. Default is an empty list.
        classes_names (list): A list of class names corresponding to the class IDs. Default is an empty list.
        masks (list): A list of masks. Default is an empty list.
        segment (bool): Whether to perform instance segmentation. Default is False.
        show_boxes (bool): Whether to show bounding boxes. Default is True.
        show_class (bool): Whether to show class labels. Default is True.
        fill_mask (bool): Whether to fill the segmented regions with color. Default is False.
        alpha (float): The transparency of filled masks. Default is 0.3.
        color_class_background (tuple): The background bgr color for class labels. Default is (0, 0, 255) (red).
        color_class_text (tuple): The text color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualization size (plot is bigger when dpi is higher). Default is 150.
        random_object_colors (bool): If true, colors for each object are selected randomly. Default is False.
        show_confidences (bool): If true and show_class=True, confidences near class are visualized. Default is False.
        axis_off (bool): If true, axis is turned off in the final visualization. Default is True.
        show_classes_list (list): If empty, visualize all classes. Otherwise, visualize only classes in the list.
        return_image_array (bool): If True, the function returns the image bgr array instead of displaying it. 
                                   Default is False.
                                   
    Returns:
        None/np.array
    """

    # Create a copy of the input image
    labeled_image = img.copy()

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
        plt.savefig('foo.png', dpi=500, bbox_inches='tight')

element_crops = MakeCropsDetectThem(
    image=img,
    model_path=yolov8_model_path,
    segment=True,
    show_crops=True,
    shape_x=340,
    shape_y=299,
    overlap_x=0,
    overlap_y=0,
    classes_list=[0],
    resize_initial_size=True,
    memory_optimize=False
)

result = CombineDetections(element_crops, match_metric='IOS')

print('YOLO-Patch-Based-Inference:')
# Final Results:
img_out=result.image
confidences=result.filtered_confidences
boxes=result.filtered_boxes
polygons=result.filtered_polygons
classes_ids=result.filtered_classes_id
classes_names=result.filtered_classes_names

# Visualizing the results using the visualize_results function
visualize_results(
    img=result.image,
    show_boxes=False,
    show_class=False,
    confidences=result.filtered_confidences,
    boxes=result.filtered_boxes,
    polygons=result.filtered_polygons,
    masks=result.filtered_masks,
    classes_ids=result.filtered_classes_id,
    classes_names=result.filtered_classes_names,
    segment=True,
)