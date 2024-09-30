[![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

# MitoHub

MitoHub is an easy-to-use mitochondrial segmentation and mobility estimation toolbox that uses state-of-the-art YOLOv9e and YOLOv8x models for segmentation and Lukas Kanade Optical flow estimation for mobility quantification using live cell microscopy images. This repository contains source code used for training and validating the neural network-based segmentation as well as information about the methodology used.

## Installation
- `git clone https://github.com/kleele-lab/MitoHub && cd MitoHub`

In case if git clone does not download the trained models, please download the trained models using [this link](https://polybox.ethz.ch/index.php/s/iSLoxOQ3TCvnnwi) and extract the `MitoHub_models.zip` to the `MitoHub_models/segmentation_models` folder:
- `conda env create -f environment.yaml`
- `conda activate mitohub`
- In case if GPU is available, install PyTorch GPU version using `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118` otherwise just proceed to the next step.
- `pip install --upgrade ultralytics`
- `pip install -r requirements.txt`
- `streamlit run Mitochondrial_Mobility_Estimation.py`

## Usage
Make sure to copy the image/video inside the cloned `MitoHub folder`.

## Mitochondrial Mobility Estimation
The web application also has a mobility detection utility which makes use of the [Shi-Tomasi Corner detection method](https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html) to extract [Lukas-Kanade optical flow](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method) in a given set of live-cell images. It further also provides a frame-by-frame mobility plot to also identify at what time-point maximum mobility was observed within these images.

Here, the user can define 
- The number of corner points (or Mitochondrial edges) to track for motion, in order to estimate the mobility of mitocondria under different conditions/treatments. 

The output from the Mobility estimation are saved in the `<User Defined Output Folder Name>_<Date_Time>/`. The output files are as follows:
- `sparse_csv_<input filename>.csv`: The format of this CSV files contains columns as `X1, Y1, X2, Y2, R, θ, ...` for each of the corner point that was detected. The index are the frame numbers, `(X1, Y1)` are the coordinates of the corner point in the preceeding frame, `(X2, Y2)` are the coordinates of the corner point in the succeeding frame and `R` is the resultant vector which determines the magnitude of motion that took place between the two frames for this particular point. We further also calculate `θ` or `theta` which is the angle for `R`. 

- `R` or `Resultant vector` is calculated as: 
```math
R =\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
```
- `θ` or `Theta` is calculated as:
```math
{θ} = {{tan^{-1}(y_2 - y_1)/(x_2 - x_1)+360}\over{360}}
```

- `sparse_flow_<input filename>.mp4`: The output video with the points and resultant vector plotted on top of it.
- `mean_mobility_<input filename>.csv`: Contains rows containing only the `R` values as well as a row-wise mean for `R` from the `sparse_csv_<input filename>.csv` to find the mean mobility across each frame.
- `optimized_<input filename>.mp4`: Optimized visualization file (using FFmpeg) used for visualization in the web application.

The output visualization for mobility estimation can be seen as follows:

<div align="center">
<video src = "https://github.com/kleele-lab/MitoHub/assets/11680352/d182d39c-6d92-4f8f-814d-cb0eb2a125fa"/>
</div>

## Mitochondrial Segmentation
In order to carry out Mitochondrial segmentation, we trained a custom [YOLOv8x-segmentation model](https://docs.ultralytics.com/tasks/segment/) achieving a MAP@50 score of 92.4% at 500 epochs. For better training and prediction, we converted large microscopy images into smaller patches and input it to the neural network similar to the approach taken in [this paper](https://arxiv.org/abs/2202.06934). Using this, we are able to achieve finer segmentation over large microscopy image files. This provides us with accurate estimation of mitochondrial area as well as mask to remove non-specific artifacts from the image.

Here, the user can define:
- The confidence threshold at which the segmentation predictions are to be made by the YOLOv8x-segmentation model (Ideally a confidence of 0.1-0.5 is good which represents 10% to 50%).
- The amount of overlap that each of the patches must have (ideally a value of 0, 1 or 2 would be good as it represnts no overlap, 1% overlap and 2% overlap between the patches)

The output from the segmentation prediction are saved in `<User Defined Output Folder Name>_<Date_Time>`. The output files are as follow:
- `mask_<input filename>.png`: Output PNG mask file that can be used for visualization
- `mask_<input filename>.tif`: Full resolution TIF mask file that can be imported to [ImageJ](https://imagej.net/ij/) for further analysis
- `patches_<input filename>.png`: A visualization of the patches that were generated for the input image
- `segment_mask_<input filename>.png`: A visualization of segmented input image.

An output visualization for the segmentation can be seen as follows:

![segment_mask_231206_RPE1_TMRe_Torin_13_MMStack_Default_decon](https://github.com/kleele-lab/MitoHub/blob/main/output_samples/segment_2024_06_25_14_16_06/segment_mask_231206_RPE1_TMRe_Torin_13_MMStack_Default_decon.png)

### Re-Training/Fine-Tuning YOLO Segmentation model
In order to carry out re-training or finetuning of the yolo segmentation neural network model that is used in MitoHub, the following steps can be carried out:
- Preprocess the images by converting them into patches using `utils/patchify_run.py` script and changing the path to input masks and image folders.
- Generate mask labels using `utils/mask_to_labels.py` script by defining the path to the mask folder.
- Transfer the patches and mask labels `.txt` files to a folder with the following folder structure and put it inside the datasets folder:
```
dataset_name/
    |-- images/
        |-- train/
        |   |-- img1.jpg
        |   |-- img2.jpg
        |   |-- ...
        |
        |-- val/
        |   |-- img3.jpg
        |   |-- img4.jpg
        |   |-- ...
        |-- ...
    |-- labels/
        |-- train/
        |   |-- mask_img1.txt
        |   |-- masK_img2.txt
        |   |-- ...
        |
        |-- val/
        |   |-- mask_img3.txt
        |   |-- mask_img4.txt
        |   |-- ...
        |-- ...

```
- Add the name of the datasets folder to `utils/dataset_seg.yaml` file.
- Run `utils/runner_segment_yolo.py` by adding the path to the `dataset_seg.yaml` file and the `model` to be fine-tuned or the `yolo` model to be used for retraining.
- The output trained model will be saved in `runs/segment/`. 
- These can then be used for performing inference by simply transfering the `runs/segment/<run_name>/weights/best.pt` or `runs/segment/<run_name>/weights/last.pt` files into the `MitoHub_models/segmentation_models/` folder.

## Further Training & Other Utilities
In order to achieve better training and predictions, the models can be further fine-tuned over more dataset. In order to to carry out pre-processing of image files, the scripts in the `utils` can be used. Please find further details on how to use them at `utils/README.md`.

Details about the results that were obtained from the current model can be found in `results` folder.
