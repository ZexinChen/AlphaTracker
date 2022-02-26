# 01 Tracking

## Preparation

Download the AlphaTracker repository. Once downloaded, change the name of the main folder from `AlphaTracker-master` to `AlphaTracker`. 

### Install Conda

This project is tested in conda env in linux, and thus that is the recommended environment. To install conda, please follow the instructions from the [conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) With conda installed, please set up the environment with the following steps.

### NVIDIA driver

Please makse sure that your NVIDIA driver version  >= 450.

### Install AlphaTracker

In your command window, locate the terminal prompt. Open this application. Then, find the folder that contains the `AlphaTracker` repository that you just downloaded. Then inside the terminal window, change the directory as follows: `cd /path/to/AlphaTracker`. 

Then run the following command:
```bash
bash install.sh
```

<br>

## Training (Optional)

We have provided pretrained models. However, if you want to train your own models on your custom dataset, you can refer to the following steps.

### Step 1. Data Preparation

Labeled data is required to train the model. The code would read RGB images and json files of
annotations to train the model. Our code is compatible with data annotated by the open source tool Sloth.
Figure 1 shows an example of annotation json file. In this example, there only two images. Each image has two mice and each mouse has two keypoint annotated.
<div align="center">
    <img src="media/jsonFormatForTraining.png", width="500" alt><br>
    Figure 1. Example of Annotation Json File
</div>

**Note** that point order matters. You must annotate all body parts in the same order for all frames. For
example, all the first points represent the nose, all the second points represent the tail and etc.
If the keypoint is not visible in one frame, then make the x,y of the keypoint to be -1.

### Step 2. Configuration

Before training, you need to charge the parameters in [Tracking/AlphaTracker/setting.py](../Tracking/AlphaTracker/setting.py) (red block in Figure 2). The meaning of the parameters can be found in the comments.
<div align="center">
    <img src="media/parameterForTracking.png", width="500" alt><br>
    Figure 2. Parameters
</div>

### Step 3. Running the code

Change directory to the [alphatracker folder](../Tracking/AlphaTracker/) and use the following command line to train the model:
```bash
# if your current virtual environment is not alphatracker
# run this command first: conda activate alphatracker
python train.py
```

<br>

### Demo data for training

If you want to test AlphaTracker's training without annotating your own data, here we provide 600 frames of two unmarked mice interacting in a homecage annotated:

https://drive.google.com/file/d/1TYIXYYIkDDQQ6KRPqforrup_rtS0YetR/view?usp=sharing

### Demo weights for tracking 

There is a demo video in [Tracking/Alphatracker/data](../Tracking/Alphatracker/data) that you can use for tracking. If you want to use the trained network we provide to track this video set `exp_name=demo` in the [Tracking/AlphaTracker/setting.py](../Tracking/AlphaTracker/setting.py)

## Tracking

### Step 1. Configuration

Before tracking, you need to change the parameters in [Tracking/AlphaTracker/setting.py](../Tracking/AlphaTracker/setting.py) (blue block in Figure 2). The meaning of
the parameters can be found in the comments.

We will use a trained weight to track a demo video by default.

### Step 2. Running the code

Change directory to the [alphatracker folder](../Tracking/AlphaTracker/) and run the following command line to do tracking:
```bash
# if your current virtual environment is not alphatracker
# run this command first: conda activate alphatracker
python track.py
```



<br>

### General Notes about the Parameters:
1. Remember not to include any spaces or parentheses in your file names. Also, file names are case-sensitive. 
2. For training the parameter num_mouse must include the same number of items as the number of json files
that have annotated data. For example if you have one json file with annotated data for 3 animals then
```num_mouse=[3]``` if you have two json files with annoted data for 3 animals then ```num_mouse=[3,3]```.
3. ```sppe_lr``` is the learning rate for the SAPE network. If your network is not performing well you can lower this
number and try retraining
4. ```sppe_epoch``` is the number of training epochs that the SAPE network does. More epochs will take longer but
can potentially lead to better performance.

<br>


