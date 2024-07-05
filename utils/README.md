## Segmentation
For the case of segmentation you can find the following scripts in this folder:

- `image_gen.py` - Convert `.h5` image files into `.png` image files that can be used for training and prediction.
- `czi_to_image.py` - Converts an input `.czi` file into `.png` image or image sequences that can be used either for segmentation or mobility detection.
- `image_gen_mask.py` - Convert `.h5` mask files into `.png` mask files that can be used for training and prediction.
- `patchify_run.py` - Convert the image and mask `.png` files into patches of specific dimensions.
- `mask_to_labels.py` - Generate polygon files from `.png` mask files.
- `runner_segment_yolov8x.py` - Train YOLOv8x segmentation neural network model for the given dataset. Make sure to modify the `datset_seg.yaml` file in order to specify the path for the generated dataset
- `predict_segmentation.py` - Run segmentation inference over custom datasets as well as over an entire folder using the trained models.
- `patch_seg.py` - Run Segmentation inference over custom dataset by dividing it firstly into patches and running the neural network inference model and later on combining the patches to obtain the final image.

Folder structure to be followed for training the neural network is as follows:

```
dataset_name/
    |-- images/
        |-- train/
        |   |-- img1.jpg
        |   |-- img2.jpg
        |   |-- ...
        |
        |-- val/
        |   |-- img1.jpg
        |   |-- img2.jpg
        |   |-- ...
        |-- ...
    |-- labels/
        |-- train/
        |   |-- img1.txt
        |   |-- img2.txt
        |   |-- ...
        |
        |-- val/
        |   |-- img1.txt
        |   |-- img2.txt
        |   |-- ...
        |-- ...

```