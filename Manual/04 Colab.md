# 04 Training and Tracking with Colab

# Introduction

AlphaTracker is tested in Linux systems broadly and those systems that have GPUs. However, if you would do not have Linux-compatible system or do not have access to consumer GPUs, you can use Google Colab for the purposes of training and tracking Alphatracker!

## An Important Notice Before You Continue:

Google Colab is only useful for training and inference purposes. If you have a fully-labeled set of data using the Alphatracker annotation tool, continue on...if you have NOT labeled your data, **please complete this step first!**

Also, if you are trying out Alphatracker and would like to use the sample data, click [here](https://drive.google.com/drive/folders/1Dk6e7sJ-dtT3L26r2Tw2QeiQSkn1DAfs?usp=sharing) to download this data. 

To access the Colab notebook, **click [here](https://colab.research.google.com/drive/1MVyZE73jzOI7bILU9vQttOpU3abK4daa?usp=sharing)**

<br>

## Step 1:

Find your folder/folders of data that contain the training images and the JSON annotated files. 

Open your Google Drive, and upload the folder/folders, and the videos you want to label, into the Drive under the main `My Drive` folder. Ensure that your Drive has ample storage after uploading the files!!

<br>

## Step 2: 

Open the `AlphaTrackerCOLAB.ipynb`, which can be found at this [link here](https://colab.research.google.com/drive/1MVyZE73jzOI7bILU9vQttOpU3abK4daa?usp=sharing). You have now opened your Python session

<br>

## Step 3:

Click `Runtime` and then `Change runtime type`...from the dropdown menu, select `GPU`. 

<p align = 'center'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/runtime_pic.png' widht = 250 height = 250>
</p>

<br>

## Step 4: 

Now, we will connect our Python session to our Google Drive. Press the play button next to the code block to run the first cell. You will be prompted with an authorization link. Click this link and follow the instructions...select the Google Drive account to which you have uploaded your data from `Step 1`. Paste the authorization code back into the prompt box. 

Run the second code block...**this is an important note**: The main `My Drive` folder has the following path: `/content/drive/My Drive`. We are now inside this main folder

<p align = 'center'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/step_4_.png'>
</p>

<br>

## Step 5:

Run the following code block to download `Alphatracker` into your Google Drive...Wait a minute or two, then go to your `My Drive` folder...you should notice a new folder by the name `Alphatracker` has appeared. 

#### **THIS FOLLOWING IS CRUCIAL:**

Go into the newly created `Alphatracker` folder. Find the subfolder entitled `01 Tracking`. **Rename this folder to `01_Tracking`...add an underscore between the "01" and "Tracking"**

<p align = 'center'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/step_5_.png'>
</p>

<br>

## Step 6:

In the following code block, in the variable `image_root_list`, enter the path to all your image folders...

In the variable `json_path_list`, enter the path to all the respective JSON training files in the same order as the image folders in `image_root_list`. 

In the variable `extension`, type in the filetype of the images...for example, `extension = 'jpg'` for JPG images.

**Remember, the main `My Drive` folder has the path `/content/drive/My Drive`...therefore, a folder named `Images` that is under the main `My Drive` folder would have the path `/content/drive/My Drive/Images`. This is what you should type. For example, `image_root_list = ['/content/drive/My Drive/Folder1', '/content/drive/My Drive/Folder2']` and etc...**

Now, navigate back to the `My Drive` folder...you will see a new folder created beginning with `TRAINING_DATA` followed by the date. This is your new training data folder that contains all the images and JSON files in the appropriate format for Colab to use...At this point, you can remove the original image folders, but not the videos, you uploaded into your Google Drive to free up some memory. 

<p align = 'center'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/step_6_.png' >
</p>

<br>

## Step 7: 

Now, find the `setting.py` file inside the `Alphatracker` directory: The file should be located in `/Alphatracker/01_Tracking/Alphatracker/setting.py`.

**Open this file using the `Text Editor` and NOT Google Docs!!**

Inside the `setting.py` file, you will find some variable that must be adjusted.

First, find the `gpu_id` variable and set it equal to `0`. For example, `gpu_id = 0`. 

Next, find the `Alphatracker_root` variable, and **copy paste the following:** `/gdrive/AlphaTracker/01_Tracking/AlphaTracker`

Next, find the `image_root_list` variable...replace it with the path to your new training data folder that was created in `Step 6`. **IMPORTANT: instead of typing `/content/drive/My Drive/TRAINING_DATA...`, type the following: `/gdrive/TRAINING_DATA...`. Essentially, replace `/content/drive/My Drive` with `/gdrive` while keeping everything else the same**

Next, find the `json_file_list` variable and do the same as above, but this time, linking to the JSON file inside the `TRAINING_DATA...` folder. 

In `num_mouse`, type in the number of animals in the training data, like the following: `num_mouse = [2]` if there are 2 animals. 

In `exp_name`, choose an experiment name and type it within the parentheses...for example, `exp_name = 'Test_Experiment'`

In `num_pose`, enter the number of poses being tracked per subject...for example, `num_pose = 4`. 

In `image_suffix`, type in the filetype in the parentheses, same as `Step 6`. 

In `yolo_iter`, enter the number of YOLO iterations...if you are simply trying the software out, enter a small number, for example, `yolo_iter = 700`. 

In `video_full_path`, enter the path to the video you are attempting to track: For example, if the video `vid1.mp4` is under the `My Drive` folder, then type: `video_full_path = '/gdrive/vid1.mp4`. 

In `result_folder`, enter the path for where you want to save tracking results...this will create a folder with the results...for example: `result_folder = /gdrive/tracking_results`. 

Save the `setting.py` file, and you should be good to go! In the image below, you will see a formatted version of the `setting.py` after adjustments have been made. 

<p align = 'center'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/setting_file.PNG' width = 500  height = 800>
</p>

<br>

## Step 8: 

Run the following code blocks back-to-back without any alterations...this will take about 6-10 minutes to complete!

<p align = 'left'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/step_8_.png'>
</p>

<br>

## Step 9:

Run the following code block to train Alphatracker! This step can take anywhere from 30 minutes to 6 hours...it depends on how many iterations you are training for.

<p align = 'left'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/step_9_.png'>
</p>

<br>

## Step 10:

Run the following code block to perform tracking on the videos you listed in `setting.py`. Once this step is complete, you can go to the folder you designated in the `result_folder` variable in `setting.py` to find the location of the tracked results!

<p align = 'left'>
    <img src = 'https://github.com/RuihanZhang2015/AlphaTracker/blob/master/00%20Manual/media/step_10_.png'>
</p>
