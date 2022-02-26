import os

# general setting
gpu_id = 0    # the id of gpu that will be used


# code path setting
AlphaTracker_root = './'


# data related settings
image_root_list = ['./data/sample_annotated_data/demo/']    # list of image folder paths to the RGB images for training
json_file_list = ['./data/sample_annotated_data/demo/train9.json']    # list of paths to the json files that contain labels of the images for training
num_mouse = [2]     # the number of mouse in the images in each image folder path
exp_name = 'demo'   # the name of the experiment
num_pose = 4    # number of the pose that is labeled, remember to change self.nJoints in train_sppe/src/utils/dataset/coco.py
pose_pair = [[0,1],[0,2],[0,3]]
train_val_split = 0.90    # ratio of data that used to train model, the rest will be used for validation
image_suffix = 'png'    # suffix of the image, png or jpg


# training hyperparameter setting
# Protip: if your training does not give good enough tracking you can lower lr and increase epoch number
# but lowering the lr too much can be bad for tracking quality as well. 
sppe_lr = 1e-4
sppe_epoch = 10
sppe_pretrain = ''
sppe_batchSize = 10
yolo_lr = 0.0005
yolo_iter = 20000+20000+20000  ## if use pretrained model please make sure yolo_iter to be large enough to guarantee finetune is done
yolo_pretrain = '' # './train_yolo/darknet/darknet53.conv.74'
yolo_batchSize = 4



# demo video setting
# note video_full_path is for track.py, video_paths is for track_batch.py
# video_full_path is the path to the video that will be tracked
video_full_path = './data/demo.mp4'
video_paths = [
  './data/demo.mp4',
  ]   # make sure video names are different from each other
start_frame = 0   # id of the start frame of the video
end_frame = 9737   # id of the last frame of the video
max_pid_id_setting = 2    # number of mice in the video
result_folder = './track_result/'   # path to the folder used to save the result
remove_oriFrame = False   # whether to remove the original frame that generated from video
vis_track_result = 1

# weights and match are parameter of tracking algorithm
# following setting should work fine, no need to change
weights = '0 6 0 0 0 0 '
match = 0


# the following code is for self-check and reformat
assert len(image_root_list) == len(
    json_file_list), 'the length of image_root_list and json_file_list should be the same'
for i in range(len(image_root_list)):
    image_root_list[i] = os.path.abspath(image_root_list[i])
    json_file_list[i] = os.path.abspath(json_file_list[i])
    
AlphaTracker_root = os.path.abspath(AlphaTracker_root)
result_folder = os.path.abspath(result_folder)