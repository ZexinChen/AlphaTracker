# %matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import time

import json
from tqdm import tqdm

from setting import AlphaTracker_root,\
        image_root_list,json_file_list,num_mouse,exp_name,num_pose,train_val_split,image_suffix,gpu_id,\
        sppe_lr, sppe_epoch,yolo_lr,yolo_iter,sppe_pretrain,yolo_pretrain,yolo_batchSize,sppe_batchSize

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

from google_drive_downloader import GoogleDriveDownloader as gdd
sppe_id = '1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW'
yolo_id = '1g8uJjK7EOlqrUCmjZTtCegwnNsBig6zn'

gdd.download_file_from_google_drive(file_id=sppe_id,
                                    dest_path=AlphaTracker_root + '/models/sppe/duc_se.pth')

gdd.download_file_from_google_drive(file_id=yolo_id,
                                    dest_path=AlphaTracker_root + '/train_yolo/darknet/darknet53.conv.74')

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

if not os.path.exists(sppe_root+ '/data/'):
    os.makedirs(sppe_root+ '/data/')
if not os.path.exists(sppe_root+ '/data/'+exp_name):
    os.makedirs(sppe_root+ '/data/'+exp_name)
if not os.path.exists(darknet_root+ '/data/'+exp_name):
    os.makedirs(darknet_root+ '/data/'+exp_name)
if not os.path.exists(darknet_root+ '/data/'+exp_name+'/color/'):
    os.makedirs(darknet_root+ '/data/'+exp_name+'/color/')
if not os.path.exists(AlphaTracker_root + '/data/'):
    os.makedirs(AlphaTracker_root + '/data/')
if not os.path.exists(AlphaTracker_root + '/data/'+exp_name):
    os.makedirs(AlphaTracker_root + '/data/'+exp_name)
if not os.path.exists(valid_image_root):
    os.makedirs(valid_image_root)
if not os.path.exists(ln_image_dir):
    os.makedirs(ln_image_dir)

## evalation setting
gt_json_file_train = AlphaTracker_root + '/data/'+exp_name+'_gt_forEval_train.json'
gt_json_file_valid = AlphaTracker_root + '/data/'+exp_name+'_gt_forEval_valid.json'
if not os.path.exists(AlphaTracker_root + '/data/'+exp_name):
    os.makedirs(AlphaTracker_root + '/data/'+exp_name)



## copy images over
import shutil
dest1 = darknet_root + '/data/' + exp_name + '/color/'
dirs = os.listdir(image_root_list[0])
print("Copying images...")
for f in dirs:
  fullpath = os.path.join(image_root_list[0], f)
  if os.path.isfile(fullpath):
    shutil.copy(fullpath, dest1)


## load and clean data
print('*** loading and clean data from json ***')
import data_utilsCOLAB
import data_utils
from imp import reload
reload(data_utilsCOLAB)

train_data,valid_data,num_allAnnot_train,num_allAnnot_valid = data_utilsCOLAB.merge_clean_ln_split_Data(image_root_list,json_file_list,ln_image_dir,train_val_split,num_mouse,num_pose,valid_image_root)

valid_len_train = len(train_data)
valid_len_valid = len(valid_data)
print('total training data len:',valid_len_train)
print('total validation data len:',valid_len_valid)

print('')
print(train_data)

print('generating data for training SPPE')

data_utilsCOLAB.generate_h5(train_h5_file,train_data,num_allAnnot=num_allAnnot_train,num_pose=num_pose,num_mouse=num_mouse)
data_utilsCOLAB.generate_h5( val_h5_file, valid_data,num_allAnnot=num_allAnnot_valid,num_pose=num_pose,num_mouse=num_mouse)

print('training h5 file is saved as:')
print(' ',train_h5_file)
print('valid h5 file is saved as:')
print(' ',val_h5_file)
print('')



print('generating data for training YOLO')


data_utilsCOLAB.generate_yolo_data(list_file=train_list_file,\
                   data_in=train_data,\
                   #image_root_in=ln_image_dir,\
                   image_root_in=image_root_list[0],\
                   yolo_annot_root=yolo_image_annot_root,\
                   image_suffix=image_suffix,\
                   color_img_prefix=color_img_prefix)
data_utilsCOLAB.generate_yolo_data(list_file=val_list_file,\
                   data_in=valid_data,\
                   #image_root_in=ln_image_dir,\
                   image_root_in=image_root_list[0],\
                   yolo_annot_root=yolo_image_annot_root,\
                   image_suffix=image_suffix,\
                   color_img_prefix=color_img_prefix)
os.system('ln -s {}/* {}/'.format(ln_image_dir,yolo_image_annot_root))

print('note that:')
print('darknet_root is: ',darknet_root)
print('new image and label are in:',yolo_image_annot_root)
print('valid_image_root is :',valid_image_root)
print('you can set *.data as follow:')
print('train  = ',train_list_file)
print('valid_data  = ',val_list_file)


print('')



    

# print('you can run the following cmd to train sppe:')
print('*** training sppe ***')
if sppe_pretrain == '':
  sppe_pretrain = '{}/models/sppe/duc_se.pth'.format(AlphaTracker_root)
sppe_train_cmd = 'CUDA_VISIBLE_DEVICES={} python train.py \\\n \
            --dataset coco \\\n \
            --img_folder_train {} \\\n \
            --annot_file_train {} \\\n \
            --img_folder_val {} \\\n \
            --annot_file_val {} \\\n \
            --expID {} \\\n \
            --nClasses {} \\\n \
            --LR {} --trainBatch {} \\\n \
            --nEpochs {} \\\n \
            --nThreads 6 \\\n \
            --loadModel {}'.format(\
                                   gpu_id, \
                                   yolo_image_annot_root,\
                                   train_h5_file,\
                                   yolo_image_annot_root,\
                                   val_h5_file,exp_name,\
                                   num_pose,\
                                   sppe_lr, sppe_batchSize,\
                                   sppe_epoch, \
                                   sppe_pretrain)

print('training with following setting:\n%s'%(sppe_train_cmd))
with cd(sppe_root+'/src'):
    os.system(sppe_train_cmd)





print('*** training detector ***')

if not os.path.exists(darknet_root+'/backup/'+exp_name):
    os.makedirs(darknet_root+'backup/'+exp_name)
if yolo_pretrain == '':
  yolo_pretrain = 'darknet53.conv.74'
## configuring mice.data
f_id = open(darknet_root+'/cfg/mice.data','w')
f_id.write('classes= 1\n')
f_id.write('train=%s \n'%(train_list_file))
f_id.write('valid=%s \n'%(val_list_file))
f_id.write('backup=backup/%s \n'%(exp_name))
f_id.write('names = data/mice.names\n')
f_id.close()
## configuring yolov3-mice.cfg
f_yolo_ori=open(darknet_root+'/cfg/yolov3-mice-ori.cfg','r+')
yolo_setting=f_yolo_ori.readlines()
f_yolo_ori.close()
yolo_setting[2]='batch=%d\n'%(yolo_batchSize)
yolo_setting[17]='learning_rate=%f\n'%(yolo_lr)
yolo_setting[19]='max_batches = %d\n'%(yolo_iter)
f_yolo=open(darknet_root+'/cfg/yolov3-mice.cfg','w+')
f_yolo.writelines(yolo_setting)
f_yolo.close()

yolo_train_cmd = './darknet detector train cfg/mice.data cfg/yolov3-mice.cfg {} -gpus {}\n'.format(yolo_pretrain, gpu_id)
f_cmd_id = open(darknet_root+'/train.sh','w')
f_cmd_id.write(yolo_train_cmd)
f_cmd_id.close()

# print('training with following setting:\ndir:%s\ncmd:%s'%(darknet_root,yolo_train_cmd))

with cd(darknet_root):
    os.system('bash train.sh')

print('training finished.')













