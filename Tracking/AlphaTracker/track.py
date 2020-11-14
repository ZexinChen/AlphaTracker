# %matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import time

import json
# import h5py
from tqdm import tqdm

import os 
import subprocess
from subprocess import Popen


from setting import AlphaTracker_root,\
        image_root_list,json_file_list,num_mouse,exp_name,num_pose,train_val_split,image_suffix,gpu_id,sppe_epoch,\
        video_full_path,start_frame,end_frame,max_pid_id_setting,result_folder,weights,match,remove_oriFrame, vis_track_result



######################################################################
###                    demo video setting                          ###
######################################################################

# video_image_save_path = result_folder + '/oriFrameFromVideo/'
video_image_save_path_base = result_folder + '/oriFrameFromVideo/'
save_image_format = "frame_%d.png"


###################################
###    code path setting        ###
###################################
darknet_root= AlphaTracker_root+'/train_yolo/darknet/'
sppe_root = AlphaTracker_root+'/train_sppe/'


###################################
###    automatic setting       ###
###################################
## general data setting
ln_image_dir = AlphaTracker_root + '/data/'+exp_name+'/color_image/'

### sppe data setting
train_h5_file = sppe_root+ '/data/'+exp_name+'/data_newLabeled_01_train.h5'
val_h5_file = sppe_root+ '/data/'+exp_name+'/data_newLabeled_01_val.h5'

### yolo data setting

color_img_prefix = 'data/'+exp_name+'/color/'
file_list_root = 'data/'+exp_name+'/'

# newdir=darknet_root + '/'+ color_img_prefix
yolo_image_annot_root =darknet_root + '/'+ color_img_prefix
train_list_file = darknet_root+'/' + file_list_root + '/' + 'train.txt'
val_list_file = darknet_root+'/' + file_list_root + '/' + 'valid.txt'

valid_image_root = darknet_root+ '/data/'+exp_name+'/valid_image/'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)




#########################################################################################################
###                                              running                                              ###
#########################################################################################################
video_image_save_path = video_image_save_path_base + '/' + video_full_path.split('/')[-1].split('.')[0]+'/frame_folder/'
print('Frame will be saved in %s'%(video_image_save_path))
print('extracting frames from video...')
## video_image_save_path = AlphaTracker_root+'/examples/'+exp_name+'_oriFrameFromVideo'
if not os.path.exists(video_image_save_path):
    os.makedirs(video_image_save_path)
else:
    os.system('rm -r {}/*'.format(video_image_save_path))

print('processing %s'%(video_full_path))    
cap = cv2.VideoCapture(video_full_path)
if cap.isOpened():
    success = True
else:
    success = False
    print(" read failed!make sure that the video format is supported by cv2.VideoCapture")

# while(success):
for frame_index in tqdm(range(end_frame)):
    success, frame = cap.read()
    if not success:
        print('read frame failed!')
        break
    if frame_index < start_frame:
        continue
    cv2.imwrite(video_image_save_path +  save_image_format % frame_index, frame)
    
cap.release()

# yolov3-mice_10000.weights
# mice_final

print('getting demo image:')
os.system('cd {}'.format(AlphaTracker_root))
demo_cmd = 'CUDA_VISIBLE_DEVICES=\'{}\' python3 demo.py \\\n \
--nClasses {} \\\n \
--indir {} \\\n \
--outdir {}  \\\n \
--yolo_model_path {}/backup/{}/yolov3-mice_final.weights \\\n \
--yolo_model_cfg {}/cfg/yolov3-mice.cfg \\\n \
--pose_model_path {}exp/coco/{}/model_{}.pkl \\\n \
--use_boxGT 0'.format(gpu_id,
    num_pose, \
	video_image_save_path,\
	result_folder,\
	darknet_root,exp_name,\
	darknet_root,\
	sppe_root,exp_name,sppe_epoch)
print(demo_cmd)
os.system(demo_cmd)


print('')
if max_pid_id_setting==1:
    print('there is only one mouse, no need to do the tracking')
else:
    track_cmd = 'python ./PoseFlow/tracker-general-fixNum-newSelect-noOrb.py \\\n \
        --imgdir {} \\\n \
        --in_json {}/alphapose-results.json \\\n \
        --out_json {}/alphapose-results-forvis-tracked.json \\\n \
        --visdir {}/pose_track_vis/  --vis {}\\\n \
        --image_format {} \
        --max_pid_id_setting {} --match {}  --weights {} \\\n \
        --out_video_path {}/{}_{}_{}_{}.mp4  \
        '.format(video_image_save_path,\
        	result_folder,\
        	result_folder,\
        	result_folder, vis_track_result,\
        	'%s.png',\
        	max_pid_id_setting, match, weights, \
        	result_folder,exp_name,max_pid_id_setting, match, weights.replace(' ', ''))
    print('tracking pose:')
    print(track_cmd)
    os.system(track_cmd)
if remove_oriFrame:
	os.system('rm -r {}'.format(video_image_save_path))

