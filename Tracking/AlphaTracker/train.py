import sys
import os
import time
import subprocess
from importlib import reload

import data_utils
from setting import AlphaTracker_root, image_root_list, json_file_list, num_mouse, \
                    exp_name, num_pose, train_val_split, image_suffix, gpu_id, \
                    sppe_lr, sppe_epoch, yolo_lr, yolo_iter, sppe_pretrain, \
                    yolo_pretrain, yolo_batchSize, sppe_batchSize


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def execute(cmd):
    if isinstance(cmd, str):
        popen = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    else:
        popen = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for stdout_line in popen.stdout:
        try:
            stdout_line_str = stdout_line.decode('ascii', errors='ignore')
        except Exception as ex1:
            try:
                stdout_line_str = stdout_line.decode('utf-8')
            except Exception as ex2:
                stdout_line_str = stdout_line.decode('cp850')
        yield stdout_line_str
    for stdout_line in popen.stderr:
        try:
            stdout_line_str = stdout_line.decode('ascii', errors='ignore')
        except Exception as ex1:
            try:
                stdout_line_str = stdout_line.decode('utf-8')
            except Exception as ex2:
                stdout_line_str = stdout_line.decode('cp850')
        yield stdout_line_str
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def executeAndWriteFile(cmd, filePath):
    try:
        with open(filePath, 'a') as fout:
            for outLine in execute(cmd):
                print(outLine)
                fout.write(outLine)
        return 0
    except subprocess.CalledProcessError:
        return 1
    else:
        return 2


# code path setting
darknet_root = AlphaTracker_root + '/train_yolo/darknet/'
sppe_root = AlphaTracker_root + '/train_sppe/'


# automatic setting
# general data setting
ln_image_dir = AlphaTracker_root + '/data/' + exp_name + '/color_image/'

# sppe data setting
train_h5_file = sppe_root + '/data/' + exp_name + '/data_newLabeled_01_train.h5'
val_h5_file = sppe_root + '/data/' + exp_name + '/data_newLabeled_01_val.h5'

# yolo data setting
color_img_prefix = 'data/' + exp_name + '/color/'
file_list_root = 'data/' + exp_name + '/'
yolo_image_annot_root = darknet_root + '/' + color_img_prefix
train_list_file = darknet_root + '/' + file_list_root + '/' + 'train.txt'
val_list_file = darknet_root + '/' + file_list_root + '/' + 'valid.txt'

valid_image_root = darknet_root + '/data/' + exp_name+'/valid_image/'

if not os.path.exists(sppe_root + '/data/'):
    os.makedirs(sppe_root + '/data/')
if not os.path.exists(sppe_root + '/data/' + exp_name):
    os.makedirs(sppe_root + '/data/' + exp_name)
if not os.path.exists(darknet_root + '/data/' + exp_name):
    os.makedirs(darknet_root + '/data/' + exp_name)
if not os.path.exists(darknet_root + '/data/' + exp_name + '/color/'):
    os.makedirs(darknet_root + '/data/' + exp_name + '/color/')
if not os.path.exists(AlphaTracker_root + '/data/'):
    os.makedirs(AlphaTracker_root + '/data/')
if not os.path.exists(AlphaTracker_root + '/data/' + exp_name):
    os.makedirs(AlphaTracker_root + '/data/' + exp_name)
if not os.path.exists(valid_image_root):
    os.makedirs(valid_image_root)
if not os.path.exists(ln_image_dir):
    os.makedirs(ln_image_dir)

# evalation setting
gt_json_file_train = AlphaTracker_root + \
    '/data/' + exp_name + '_gt_forEval_train.json'
gt_json_file_valid = AlphaTracker_root + \
    '/data/' + exp_name + '_gt_forEval_valid.json'
if not os.path.exists(AlphaTracker_root + '/data/' + exp_name):
    os.makedirs(AlphaTracker_root + '/data/' + exp_name)


# data preparation
reload(data_utils)
train_data, valid_data, num_allAnnot_train, num_allAnnot_valid = data_utils.merge_clean_ln_split_Data(
    image_root_list, json_file_list, ln_image_dir, train_val_split, num_mouse, num_pose, valid_image_root)
valid_len_train = len(train_data)
valid_len_valid = len(valid_data)
print('total training data:', valid_len_train)
print('total validation data:', valid_len_valid)


print('\ngenerating data for training pose estimation')
data_utils.generate_h5(train_h5_file, train_data,
                       num_allAnnot=num_allAnnot_train, num_pose=num_pose, num_mouse=num_mouse)
data_utils.generate_h5(val_h5_file, valid_data,
                       num_allAnnot=num_allAnnot_valid, num_pose=num_pose, num_mouse=num_mouse)
print('training h5 file is saved as:', train_h5_file)
print('valid h5 file is saved as:', val_h5_file)


print('\ngenerating data for training YOLO')
data_utils.generate_yolo_data(list_file=train_list_file,
                              data_in=train_data,
                              image_root_in=ln_image_dir,
                              yolo_annot_root=yolo_image_annot_root,
                              image_suffix=image_suffix,
                              color_img_prefix=color_img_prefix)
data_utils.generate_yolo_data(list_file=val_list_file,
                              data_in=valid_data,
                              image_root_in=ln_image_dir,
                              yolo_annot_root=yolo_image_annot_root,
                              image_suffix=image_suffix,
                              color_img_prefix=color_img_prefix)
os.system('ln -s {}/* {}/'.format(ln_image_dir, yolo_image_annot_root))


print('\n\n*** training detector ***')

if not os.path.exists(darknet_root+'/backup/'+exp_name):
    os.makedirs(darknet_root+'backup/'+exp_name)
if yolo_pretrain == '':
    yolo_pretrain = 'darknet53.conv.74'

# configuring mice.data
try:
    os.system('mkdir {}/cfg'.format(darknet_root))
except Exception as e:
    print(e)
f_id = open(darknet_root + '/cfg/mice.data', 'w')
f_id.write('classes = 1\n')
f_id.write('train = %s \n' % (train_list_file))
f_id.write('valid = %s \n' % (val_list_file))
f_id.write('backup = backup/%s \n' % (exp_name))
f_id.write('names = data/mice.names\n')
f_id.close()

# configuring yolov3-mice.cfg
f_yolo_ori = open(darknet_root + '/cfg/yolov3-mice-ori.cfg', 'r+')
yolo_setting = f_yolo_ori.readlines()
f_yolo_ori.close()
yolo_setting[2] = 'batch = %d\n' % (yolo_batchSize)
yolo_setting[17] = 'learning_rate = %f\n' % (yolo_lr)
yolo_setting[19] = 'max_batches = %d\n' % (yolo_iter)
f_yolo = open(darknet_root + '/cfg/yolov3-mice.cfg', 'w+')
f_yolo.writelines(yolo_setting)
f_yolo.close()

yolo_train_cmd = './darknet detector train cfg/mice.data cfg/yolov3-mice.cfg {} -gpus {}\n'.format(
    yolo_pretrain, gpu_id)
f_cmd_id = open(darknet_root + '/train.sh', 'w')
f_cmd_id.write(yolo_train_cmd)
f_cmd_id.close()

time_start = time.time()
filePath = os.path.abspath('./outputOfYoloTraining.txt')
with cd(darknet_root):
    exitCode = executeAndWriteFile(['bash', 'train.sh'], filePath)
    if exitCode != 0:
        print(
            'failed to train YOLO, please check output of YOLO training in %s' % (filePath))
        sys.exit(1)
time_end = time.time()
print('YOLO training finished. Time used: {} seconds'.format(time_end - time_start))


# print('you can run the following cmd to train sppe:')
print('*** training sppe ***')
if sppe_pretrain == '':
    sppe_pretrain = '{}/models/sppe/duc_se.pth'.format(AlphaTracker_root)
sppe_pretrain = os.path.abspath(sppe_pretrain)
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
            --loadModel {}'.format(
    gpu_id,
    yolo_image_annot_root,
    train_h5_file,
    yolo_image_annot_root,
    val_h5_file, exp_name,
    num_pose,
    sppe_lr, sppe_batchSize,
    sppe_epoch,
    sppe_pretrain)

print('training with following setting:\n%s' % (sppe_train_cmd))
filePath = os.path.abspath('./outputOfSPPETraining.txt')
with cd(sppe_root + '/src'):
    # exitCode = executeAndWriteFile(sppe_train_cmd, filePath)
    # if exitCode != 0:
    #     print(
    #         'failed to train sppe, please check output of YOLO training in %s' % (filePath))
    #     sys.exit(1)
    # else:
    #     print('SPPE is trained, please check output of YOLO training in %s' % (filePath))
    os.system(sppe_train_cmd)
