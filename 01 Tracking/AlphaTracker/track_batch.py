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
import sys
import subprocess
from subprocess import Popen


from setting import AlphaTracker_root,\
        image_root_list,json_file_list,num_mouse,exp_name,num_pose,\
        train_val_split,image_suffix,gpu_id,sppe_epoch,\
        video_paths,start_frame,end_frame,max_pid_id_setting,\
        result_folder,weights,match,remove_oriFrame,vis_track_result



######################################################################
###                    demo video setting                          ###
######################################################################

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
    os.mkdir(result_folder)




#########################################################################################################
###                                              running                                              ###
#########################################################################################################
def walk_path(p,all_video_paths):
    for root, dirs, files in os.walk(p):
        for f in files:
            all_video_paths.append(os.path.join(root, f))
        for d in dirs:
            all_video_paths = walk_path(os.path.join(root, d),all_video_paths)
    return all_video_paths
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if os.path.exists(path) and os.path.isdir(path):
            pass
        else: raise

## get all paths to all the video
all_video_paths = []
for p in video_paths:
    if not os.path.exists(p):
        print('path not exists!:',p)
        continue
    if os.path.isdir(p):
        all_video_paths = walk_path(p,all_video_paths)
    else:
        all_video_paths.append(p)
## check duplicate video name
all_video_paths_noDuplicate = []
name_path_dict = {}
for p in all_video_paths:
    vn = p.split('/')[-1].split('.')[0]
    if vn in name_path_dict:
        print('Error: duplicate video name!')
        print(name_path_dict[vn])
        print(p)
        print('Only one will be tracked:')
        print(name_path_dict[vn])
    else:
        name_path_dict[vn] = p
        all_video_paths_noDuplicate.append(p)
all_video_paths = all_video_paths_noDuplicate
print('')

s_list = []
f_list = []
if (len(sys.argv)!=1):
    all_video_paths = [all_video_paths[int(sys.argv[1])]]
    print('only deal with the %d th video path: \n %s'%(int(sys.argv[1]),all_video_paths[0]))
for p in all_video_paths:
    if ('0913' in p \
        or '0902_black_two' in p \
        or '0910_black_two' in p \
        or '0854_black_two' in p \
        or '1028_black_two' in p ):
        continue
    print('deal with %s'%(p))
    print('extracting frames from video...')
    ## video_image_save_path = AlphaTracker_root+'/examples/'+exp_name+'_oriFrameFromVideo'
    # if not os.path.exists(video_image_save_path):
    #     os.mkdir(video_image_save_path)
    # else:
    #     os.system('rm {}/*'.format(video_image_save_path))
    video_image_save_path = video_image_save_path_base + '/' + p.split('/')[-1].split('.')[0]+'/frame_folder/'
    mkdir_p(video_image_save_path)
    
    cap = cv2.VideoCapture(p)
    if cap.isOpened():
        success = True
        s1 = 0
    else:
        success = False
        s1 = 1
        print(" read failed!make sure that the video format is supported by cv2.VideoCapture")

    print('extracting frames to %s...'%(video_image_save_path))
    # while(success):

    for frame_index in tqdm(range(end_frame)):
        success, frame = cap.read()
        if not success:
            print('read frame failed!')
            break
        if frame_index < start_frame:
            continue
        try:
            cv2.imwrite(video_image_save_path +  save_image_format % frame_index, frame)
        except:
            print('Write image error!')
            break
        
    cap.release()

    # yolov3-mice_10000.weights
    # mice_final
    result_folder_special = result_folder + '/' + p.split('/')[-1].split('.')[0]
    mkdir_p(result_folder_special)

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
    	result_folder_special,\
    	darknet_root,exp_name,\
    	darknet_root,\
    	sppe_root,exp_name,sppe_epoch)
    print(demo_cmd)
    s2 = os.system(demo_cmd)

    print('')
    if max_pid_id_setting==1:
        print('there is only one mouse, no need to do the tracking')
    else:
        print('tracking pose:')
        cmd_line = 'python ./PoseFlow/tracker-general-fixNum-newSelect-noOrb.py \\\n \
            --imgdir {} \\\n \
            --in_json {}/alphapose-results.json \\\n \
            --out_json {}/alphapose-results-forvis-tracked.json \\\n \
            --visdir {}/pose_track_vis/ --vis {} \\\n \
            --image_format {} \
            --max_pid_id_setting {} --match {}  --weights {} \\\n \
            --out_video_path {}/{}_{}_{}_{}.mp4  \
            '.format(video_image_save_path,\
            	result_folder_special,\
            	result_folder_special,\
            	result_folder_special, vis_track_result,\
            	'%s.png',\
            	max_pid_id_setting, match, weights, \
            	result_folder_special,exp_name,max_pid_id_setting, match, weights.replace(' ', ''))
        print(cmd_line)
        s3 = os.system(cmd_line)

    if s1==0 and s2==0 and s3==0:
        s_list.append(p)
    else:
        f_list.append(p)


    if remove_oriFrame:
    	os.system('rm -r {}'.format(video_image_save_path))
print('s_list:',s_list)
print('f_list:',f_list)
print('s_list',['\''+p.split('/')[-1].split('.')[0]+'\'' for p in s_list])


