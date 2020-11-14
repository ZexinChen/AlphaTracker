
######################################################################
###                        general setting                         ###
######################################################################
# gpu_id is the id of gpu that will be used
gpu_id = 5


######################################################################
###                         code path setting                      ###
######################################################################
# AlphaTracker_root = '/disk2/zexin/project/mice/AlphaTracker/'
AlphaTracker_root = '/home/zexin/project/mice/algorithm/tracking/AlphaTracker'


######################################################################
###                        data related setting                    ###
######################################################################
# # image_root_list is a list of image folder paths to the RGB image for training
# image_root_list=[\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/', # image folder path 1
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/', # image folder path 2
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05/', # image folder path 3
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/select_track_frame_merge/', # image folder path 4
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/select_track_frame_merge/', # image folder path 4
#                 ]
# # json_file_list is a list of paths to the json file that contain labels of the images for training
# json_file_list = [\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json', # label json file 1
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json', # label json file 2
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/multi_person_05.json', # label json file 3
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/multi_person_06.json',
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/multi_person_07.json',
#                  ]
# # num_mouse is a list the specify the number of mouse in the images in each image folder path
# num_mouse = [4,4,4,4,5]
# # exp_name is the name of the experiment
# exp_name = 'labeled_byCompany_0204050607_split90_ori'
# # num_pose is the number of the pose that is labeled
# ## note!! remember to change self.nJoints in train_sppe/src/utils/dataset/coco.py
# num_pose = 4
# pose_pair = [[0,1],[0,2],[0,3]]
# # train_val_split is ratio of data that used to train model
# train_val_split = 0.90
# # image_suffix is the suffix of the image
# image_suffix = 'png'



# image_root_list is a list of image folder paths to the RGB image for training
image_root_list=[\
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/',
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05/',
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/select_track_frame_merged/',
                # '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/select_track_frame_merged/',
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/08/select_track_frame/all',

                 # '/disk2/zexin/data/mice/train_data/data_04/',
                 # '/disk2/zexin/data/mice/train_data/data_05/', # image folder path 3
                 # '/disk2/zexin/data/mice/train_data/data_06/',
                 # '/disk2/zexin/data/mice/train_data/data_07/',
                ]
# json_file_list is a list of paths to the json file that contain labels of the images for training
json_file_list = [\
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json',
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/multi_person_05.json',
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/multi_person_06.json',
                # '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/multi_person_07.json',
                '/disk4/zexin/datasets/mice/new_labeled_byCompany/08/multi_person_08.json',
                  # '/disk2/zexin/data/mice/train_data/multi_person_04.json', # label json file 3
                  # '/disk2/zexin/data/mice/train_data/multi_person_05.json', # label json file 3
                  # '/disk2/zexin/data/mice/train_data/multi_person_06.json', # label json file 3
                  # '/disk2/zexin/data/mice/train_data/multi_person_07.json', # label json file 3
                 ]

# num_mouse is a list the specify the number of mouse in the images in each image folder path
num_mouse = [4,4,4,4]
# exp_name is the name of the experiment
exp_name = 'labeled_byCompany_4567_split90_ori'
# num_pose is the number of the pose that is labeled
## note!! remember to change self.nJoints in train_sppe/src/utils/dataset/coco.py
num_pose = 4
pose_pair = [[0,1],[0,2],[0,3]]
# train_val_split is ratio of data that used to train model
train_val_split = 0.90
# image_suffix is the suffix of the image
image_suffix = 'png'



# # image_root_list is a list of image folder paths to the RGB image for training
# image_root_list=[\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/08/select_track_frame/all/', # image folder path 4
#                 ]
# # json_file_list is a list of paths to the json file that contain labels of the images for training
# json_file_list = [\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/08/multi_person_08.json',
#                  ]
# # num_mouse is a list the specify the number of mouse in the images in each image folder path
# num_mouse = [2]
# # exp_name is the name of the experiment
# exp_name = 'labeled_byCompany_08_split80_withLimbs_pretrain_correct_refine'

# # num_pose is the number of the pose that is labeled
# ## note!! remember to change self.nJoints in train_sppe/src/utils/dataset/coco.py
# num_pose = 8
# pose_pair = [[0,1],[0,2],[0,3],[4,5],[5,7],[7,6],[6,4]]
# # train_val_split is ratio of data that used to train model
# train_val_split = 0.80
# # image_suffix is the suffix of the image
# image_suffix = 'png'

######################################################################
###               training hyperparameter setting                  ###
######################################################################
sppe_lr=1e-4
sppe_epoch=10
# sppe_pretrain = '/disk4/zexin/project/mice/AlphaTracker/train_sppe/exp/coco/labeled_byCompany_0204050607_split90_ori/model_10.pkl'
sppe_pretrain = ''
sppe_batchSize = 10
yolo_lr=0.0005
yolo_iter=20000+20000+20000  ## if use pretrained model please make sure yolo_iter to be large enough to garantee finetune is done
# yolo_pretrain = '/disk4/zexin/project/mice/AlphaTracker/train_yolo/darknet/backup/labeled_byCompany_0204050607_split90_ori/yolov3-mice_final.weights'
yolo_pretrain = ''
yolo_batchSize = 4


######################################################################
###                    demo video setting                          ###
######################################################################
### note video_full_path is for track.py, video_paths is for track_batch.py
# video_full_path is the path to the video that will be tracked
# video_full_path = '/disk4/zexin/project/mice/datasets/0603/1959_black_two.mov'
# video_full_path = '/disk4/zexin/ruihan/mice/datasets/20190603/1929_black_two.mov'
# video_full_path = '/disk4/zexin/project/mice/datasets/interaction/2019-10-25_15-29-43_femalespost3hourisolation.mp4'
# video_full_path = '/disk4/zexin/project/mice/datasets/06_040506_twoMice/0854_black_two.mov'
video_full_path = '/home/zexin/project/mice/datasets/interaction/2019-11-19_2femalespost8hrisolation.mp4'
video_paths = [
	# '/disk4/zexin/ruihan/mice/datasets/0605/',
	# '/disk4/zexin/ruihan/mice/datasets/0604/',
  # '/disk4/zexin/project/mice/datasets/06_040506_twoMice',
  # '/disk4/zexin/ruihan/mice/datasets/20191030_social/',
  # '/disk4/zexin/project/mice/datasets/interaction/',
  
  '/disk2/zexin/data/mice/demo_videos/2019-11-19_2femalespost8hrisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-10-25_15-29-43_femalespost3hourisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-11-04_females4hrpostisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-11-07_femaleandmalepost6hrisolation.mp4',
  '/disk2/zexin/data/mice/demo_videos/2019-11-07_malemaleinteraction6hrpostisolation.mp4',
	] ## make sure video names are different from each other
# start_frame is the id of the start frame of the video
start_frame = 0
# end_frame is the id of the last frame of the video
end_frame = 3000000
# max_pid_id_setting is the number of mice in the video
max_pid_id_setting = 2
# result_folder is the path to the folder used to save the result
# result_folder = '/home/zexin/project/mice/AlphaTracker/examples/tracke_result_folder/'
# result_folder = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/withLimbs_interaction_refine/'
# result_folder = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/noLimbs_interaction/'
result_folder = '/disk2/zexin/project/mice/track_result/time_testing/'
# remove_oriFrame is whether remove the original frame that generated from video
remove_oriFrame = False
vis_track_result = 0
# weights and match are parameter of tracking algorithm, following setting should work fine, no need to change
weights = '0 6 0 0 0 0 '
match = 0
