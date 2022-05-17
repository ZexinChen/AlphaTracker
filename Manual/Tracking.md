# 01 Tracking

## Preparation

Download the AlphaTracker repository either by downloading the zip file or by opening a linux terminal and typing

`git clone https://github.com/ZexinChen/AlphaTracker.git`

Once it has been downloaded, change the directory:

`cd AlphaTracker`

### Install Conda

This project is tested using a conda environment in linux, and thus that is the recommended environment. To install conda, please follow the instructions from the [conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) With conda installed, please set up the environment with the following steps.

We recommend using either Anaconda3 or miniconda3. 

### NVIDIA driver

If you are using a GPU, you should already have installed the appropriate CUDA support to make use of it.  Please make sure that your NVIDIA driver version  >= 450.  You can check your driver version with the following command:

`nvidia-smi`

The top line will look  something like this:

Tue May 17 15:28:51 2022

+-----------------------------------------------------------------------------+

| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |

|-------------------------------+----------------------+----------------------+

If the Driver Version is less than 450, you will need to update your drivers before continuing with this installation.

This also will tell you your CUDA version.  Hold on to this information as you will need it in the next section.

### Install AlphaTracker

#### Update install.sh
We will need to alter the install.sh script to reflect the correct hardware for your system.  Use your favorite editor (emacs, vi, atom, etc.) to open the file install.sh which is located in the AlphaTracker folder.  

Locate line 8: 

    echo ". ~/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc

Please alter this line to use the conda version that you have installed.  If you are using anaconda3, you don't need to change this line.  However, for example, if you are using miniconda3, you would change this to read:

    echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
Next, locate line 15:

    conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
If your CUDA version is 11.1 or higher, you will not need to change this line.  If you're running CUDA version 10.2 or higher you should change the line to 

    conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch -c conda-forge -y

If your version is older, head to the [pytorch site](https://pytorch.org/get-started/locally/) to figure out which cudatoolkit matches your system.  
Make sure you save your changes to install.sh!

#### Update Makefile for CUDA versions older than 11.1
If you are running an older CUDA version, there is one more file to update before running the install.sh script.  Please open the file "AlphaTracker/Tracking/AlphaTracker/train_yolo/darknet/Makefile" using an editor.  This file will build the YOLO detector and needs to know the architecture of your GPU to run correctly.  You can check your CUDA version using the command 

`nvcc --version`

It will return something like this:

``` nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2018 NVIDIA Corporation
    Built on Sat_Aug_25_21:08:01_CDT_2018
    Cuda compilation tools, release 10.0, V10.0.130```

In this example, we are running V10.0.  So we would edit the Makefile, as per the instructions at the top of that file to remove the lines 
```   -gencode arch=compute_80,code=sm_80 \
 	  -gencode arch=compute_86,code=sm_86 ```
      
Your Makefile should now contain the (uncommented) command:

    ```ARCH= -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \      
      -gencode arch=compute_52,code=[sm_52,compute_52] \            
	  -gencode arch=compute_60,code=sm_60 \            
	  -gencode arch=compute_61,code=[sm_61,compute_61] \            
	  -gencode arch=compute_62,code=sm_62 \            
	  -gencode arch=compute_70,code=sm_70 \ 
	  -gencode arch=compute_75,code=sm_75```
      
      
Save the file.

#### Run the install.sh script to install AlphaTracker and download the sample datasets
In terminal prompt, make sure you are in the AlphaTracker folder.Then run the following command:
```bash
bash install.sh
```

Some institutions block automatic downloads of Google drive files.  You can verify that your install completed this step by checking that the model was downloaded into the correct location:

`ls -l AlphaTracker/Tracking/AlphaTracker/train_sppe/exp/coco/demo/model_10.pkl`
It will return something like:

`-rw-r--r-- 1 jb cnlOther 238755600 May 17 14:44 AlphaTracker/Tracking/AlphaTracker/train_sppe/exp/coco/demo/model_10.pkl`

Note that this is a large file. If the returned size of your file is small, then your institution is likely blocking the download.  Please follow the instructions [here](https://github.com/ZexinChen/AlphaTracker/issues/9) to manually install each file.

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


