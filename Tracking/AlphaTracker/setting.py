
######################################################################
###                        general setting                         ###
######################################################################
# gpu_id is the id of gpu that will be used
import os
gpu_id = 0


######################################################################
###                         code path setting                      ###
######################################################################
# AlphaTracker_root = '/disk2/zexin/project/mice/AlphaTracker/'
AlphaTracker_root = './'


######################################################################
###                        data related settings                    ###
######################################################################

# image_root_list is a list of image folder paths to the RGB image for training
image_root_list=[\
                './data/sample_annotated_data/demo/'
                ]
# json_file_list is a list of paths to the json file that contain labels of the images for training
json_file_list = [\
                './data/sample_annotated_data/demo/train9.json'
                ]

# num_mouse is a list the specify the number of mouse in the images in each image folder path
num_mouse = [2]
# exp_name is the name of the experiment
exp_name = 'demo'
# num_pose is the number of the pose that is labeled
## note!! remember to change self.nJoints in train_sppe/src/utils/dataset/coco.py
num_pose = 4
pose_pair = [[0,1],[0,2],[0,3]]
# train_val_split is ratio of data that used to train model
train_val_split = 0.90
# image_suffix is the suffix of the image. Accepted images are png or jpg
image_suffix = 'png'

######################################################################
###               training hyperparameter setting                  ###
######################################################################
#protip: if your training does not give good enough tracking you can lower lr and increase epoch number. But lowering the lr too much can be bad for tracking quality as well. 
sppe_lr=1e-4
sppe_epoch=10
# sppe_pretrain = '/disk4/zexin/project/mice/AlphaTracker/train_sppe/exp/coco/labeled_byCompany_0204050607_split90_ori/model_10.pkl'
sppe_pretrain = ''
sppe_batchSize = 10
yolo_lr=0.0005
yolo_iter=20000+20000+20000  ## if use pretrained model please make sure yolo_iter to be large enough to guarantee finetune is done
# yolo_pretrain = '/disk4/zexin/project/mice/AlphaTracker/train_yolo/darknet/backup/labeled_byCompany_0204050607_split90_ori/yolov3-mice_final.weights'
yolo_pretrain = ''
yolo_batchSize = 4


######################################################################
###                    demo video setting                          ###
######################################################################
### note video_full_path is for track.py, video_paths is for track_batch.py
# video_full_path is the path to the video that will be tracked

video_full_path = './data/demo.mp4'
video_paths = [
  '/disk2/zexin/data/mice/demo_videos/2019-11-19_2femalespost8hrisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-10-25_15-29-43_femalespost3hourisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-11-04_females4hrpostisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-11-07_femaleandmalepost6hrisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-11-07_malemaleinteraction6hrpostisolation.mp4',
  ] ## make sure video names are different from each other
# start_frame is the id of the start frame of the video
start_frame = 0
# end_frame is the id of the last frame of the video
end_frame = 300
# max_pid_id_setting is the number of mice in the video
max_pid_id_setting = 2
# result_folder is the path to the folder used to save the result
# result_folder = '/home/zexin/project/mice/AlphaTracker/examples/tracke_result_folder/'
# result_folder = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/withLimbs_interaction_refine/'
# result_folder = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/noLimbs_interaction/'
result_folder = './track_result/'
# remove_oriFrame is whether remove the original frame that generated from video
remove_oriFrame = False
vis_track_result = 1
# weights and match are parameter of tracking algorithm, following setting should work fine, no need to change
weights = '0 6 0 0 0 0 '
match = 0


## the following code is for self-check and reformat
assert len(image_root_list) == len(
    json_file_list), 'the length of image_root_list and json_file_list should be the same'
for i in range(len(image_root_list)):
    image_root_list[i] = os.path.abspath(image_root_list[i])
    json_file_list[i] = os.path.abspath(json_file_list[i])
AlphaTracker_root = os.path.abspath(AlphaTracker_root)
result_folder = os.path.abspath(result_folder)

