# %matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import time

import json
import h5py
from tqdm import tqdm



###################################
###    code path setting        ###
###################################
alphaPose_root = '/home/zexin/project/mice/AlphaPose/'
darknet_root= alphaPose_root+'/train_yolo/darknet/'
sppe_root = alphaPose_root+'/train_sppe/'


image_root_list=[\
                 '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04',\
                ]
json_file_list = [\
                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04_preview.json',\
                 ]
num_mouse = [4]
exp_name = 'labeled_byCompany_04pre_split90_ori'
num_pose = 4
train_val_split = 0.90

###################################
###    automatic setting       ###
###################################
## general data setting
ln_image_dir = alphaPose_root + '/data/mice/'+exp_name+'/color_image/'

### sppe data setting
train_h5_file = sppe_root+ '/data/mice/'+exp_name+'/data_newLabeled_01_train.h5'
val_h5_file = sppe_root+ '/data/mice/'+exp_name+'/data_newLabeled_01_val.h5'

### yolo data setting
image_suffix = 'png'

color_img_prefix = 'data/mice/'+exp_name+'/color/'
file_list_root = 'data/mice/'+exp_name+'/'

# newdir=darknet_root + '/'+ color_img_prefix
yolo_image_annot_root =darknet_root + '/'+ color_img_prefix
train_list_file = darknet_root+'/' + file_list_root + '/' + 'train.txt'
val_list_file = darknet_root+'/' + file_list_root + '/' + 'valid.txt'

valid_image_root = darknet_root+ '/data/mice/'+exp_name+'/valid_image/'

## evalation setting
gt_json_file_train = alphaPose_root + '/data/mice/'+exp_name+'_gt_forEval_train.json'
gt_json_file_valid = alphaPose_root + '/data/mice/'+exp_name+'_gt_forEval_valid.json'

if not os.path.exists(sppe_root+ '/data/mice/'+exp_name):
    os.mkdir(sppe_root+ '/data/mice/'+exp_name)
if not os.path.exists(darknet_root+ '/data/mice/'+exp_name):
    os.mkdir(darknet_root+ '/data/mice/'+exp_name)
if not os.path.exists(darknet_root+ '/data/mice/'+exp_name+'/color/'):
    os.mkdir(darknet_root+ '/data/mice/'+exp_name+'/color/')
if not os.path.exists(valid_image_root):
    os.mkdir(valid_image_root)
if not os.path.exists(alphaPose_root + '/data/mice/'+exp_name):
    os.mkdir(alphaPose_root + '/data/mice/'+exp_name)
if not os.path.exists(ln_image_dir):
    os.mkdir(ln_image_dir)







import sys
sys.path.append('../')
## load and clean data
import data_utils
from imp import reload
reload(data_utils)

train_data,valid_data,num_allAnnot_train,num_allAnnot_valid =  data_utils.merge_clean_ln_split_Data(image_root_list,json_file_list,ln_image_dir,train_val_split,num_mouse,num_pose,valid_image_root)

valid_len_train = len(train_data)
valid_len_valid = len(valid_data)
print(valid_len_train,valid_len_valid)



data_utils.generate_h5(train_h5_file,train_data,num_allAnnot=num_allAnnot_train,num_pose=num_pose)
data_utils.generate_h5( val_h5_file, valid_data,num_allAnnot=num_allAnnot_valid,num_pose=num_pose)

print('training h5 file is saved as:')
print(' ',train_h5_file)
print('valid h5 file is saved as:')
print(' ',val_h5_file)



with h5py.File(train_h5_file, 'r') as annot:
    imgname_coco_train = annot['imgname'][:]
    bndbox_coco_train = annot['bndbox'][:]
    part_coco_train = annot['part'][:]
    size_train = imgname_coco_train.shape[0]
    
import scipy.misc
from functools import reduce

for index in tqdm(range(size_train)):
    part = part_coco_train[index]
    bndbox = bndbox_coco_train[index]
    imgname = imgname_coco_train[index]
    
    imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
    img_path = os.path.join(ln_image_dir, imgname)
    img = scipy.misc.imread(img_path, mode='RGB')
    
    cv2.rectangle(img,(int(bndbox[0][0]),int(bndbox[0][1])),(int(bndbox[0][2]),int(bndbox[0][3])), (0,0,255), 3)
    for kk in range(len(part)):
        cv2.circle(img,(int(part[kk][0]), int(part[kk][1])),10,(255,(100*kk)%255,0),-1)
        cv2.putText(img,str(kk),(int(part[kk][0]), int(part[kk][1])),\
                                               cv2.FONT_HERSHEY_COMPLEX,1,(255,(100*kk)%255,0),3)
    
    cv2.imwrite('./vis_sppe_input/'+imgname[:-4]+str(index)+imgname[-4:],img)
    # visImage(img)
    
    # if index>5:
        # break


























