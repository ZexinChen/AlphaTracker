# %matplotlib inline



import os



def runOnOneVideo(vp,vn):
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

    #########################################################################################################
    ###                                  demo video setting                                               ###
    #########################################################################################################


    # video_full_path = '/disk4/zexin/project/mice/datasets/0603/1312_black_two.mov'
    video_full_path = vp + '/' + vn
    start_frame = 0
    end_frame = 80000
    image_folder_name = vp.split('/')[-1]+'_'+vn.split('.')[0] +'_mov_%d_%d'%(start_frame,end_frame)
    max_pid_id_setting = 2
    match = 0
    weights = '0 6 0 0 0 0 '
    # video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
    save_image_format = image_folder_name+"_%d.png"




    #########################################################################################################
    ###                                  code path setting                                                ###
    #########################################################################################################
    alphaPose_root = '/home/zexin/project/mice/AlphaPose/'
    darknet_root= alphaPose_root+'/train_yolo/darknet/'
    sppe_root = alphaPose_root+'/train_sppe/'


    #########################################################################################################
    ###                                 data related setting                                              ###
    #########################################################################################################


    image_root_list=[\
                     '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/',\
                     '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/', \
                     '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05/',\
                     '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/select_track_frame_merged/', \
                     '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/select_track_frame_merge/'
                    ]
    json_file_list = [\
                      '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json',\
                      '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json',\
                      '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/multi_person_05.json',\
                      '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/multi_person_06.json', \
                      '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/multi_person_07.json'
                     ]
    num_mouse = [4,4,4,4,5]
    exp_name = 'labeled_byCompany_0204050607_split90_ori'
    num_pose = 4
    train_val_split = 0.90


    #########################################################################################################
    ###                                    automatic setting                                              ###
    #########################################################################################################
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






    #########################################################################################################
    ###                                              running                                              ###
    #########################################################################################################

    video_image_save_path = alphaPose_root+'/examples/'+exp_name+'_'+image_folder_name+'_fromVideo'+'/oriFrameFromVideo'
    if not os.path.exists(video_image_save_path):
        os.mkdir(alphaPose_root+'/examples/'+exp_name+'_'+image_folder_name+'_fromVideo')
        os.mkdir(video_image_save_path)
    else:
        os.system('rm {}/*'.format(video_image_save_path))
        
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


    import os 
    import subprocess
    from subprocess import Popen

    print('getting demo image:')
    os.system('cd {}'.format(alphaPose_root))
    # subprocess.run('conda activate poseflow')
    # subprocess.run("bash -c 'conda activate poseflow'", shell=True)
    os.system('CUDA_VISIBLE_DEVICES=\'4\' python3 demo.py \\\n \
    --indir {} \\\n \
    --outdir examples/{}/  \\\n \
    --yolo_model_path {}/backup/{}/yolov3-mice.backup \\\n \
    --yolo_model_cfg /disk4/zexin/project/mice/AlphaPose/train_yolo/darknet//cfg/yolov3-mice.cfg \\\n \
    --pose_model_path {}exp/coco/{}/model_30.pkl \\\n \
    --use_boxGT 0'.format(video_image_save_path,exp_name+'_'+image_folder_name+'_fromVideo',darknet_root,exp_name,sppe_root,exp_name))


    print('')
    if max_pid_id_setting==1:
        print('there is only one mouse, no need to do the tracking')
    else:
        print('tracking pose:')
        # print('use http://localhost:21600/notebooks/project/mice/PoseFlow/tracker-general.ipynb')
        # print('self.imgdir = \'{}\' \n \
        # self.in_json = \'{}/examples/{}/alphapose-results.json\' \n \
        # self.out_json = \'{}/examples/{}/alphapose-results-forvis-tracked.json\' \n \
        # self.visdir = \'{}examples/{}/pose_track_vis/\'  \n \
        # self.image_format = \'{}\'  \n \
        # '.format(video_image_save_path,alphaPose_root,exp_name+'_'+image_folder_name+'_fromVideo',alphaPose_root,exp_name+'_'+image_folder_name+'_fromVideo',alphaPose_root,exp_name+'_'+image_folder_name+'_fromVideo',save_image_format))

        # print('or run the following cmd:')
        # print('cd {}/PoseFlow'.format(alphaPose_root))
        os.system('python ./PoseFlow/tracker-general-fixNum-newSelect-noOrb.py \\\n \
            --imgdir {} \\\n \
            --in_json {}/examples/{}/alphapose-results.json \\\n \
            --out_json {}/examples/{}/alphapose-results-forvis-tracked.json \\\n \
            --visdir {}/examples/{}/pose_track_vis/ \\\n \
            --image_format {} \
            --max_pid_id_setting {} --match {}  --weights {} \\\n \
            --out_video_path {}/examples/{}/{}_{}_{}_{}.mp4  \
            '.format(video_image_save_path,\
                alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',\
                alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',\
                alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',\
                '%s.png',\
                max_pid_id_setting, match, weights, \
                alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',exp_name+'_'+image_folder_name+'_fromVideo',max_pid_id_setting, match, weights.replace(' ', '')))
        #     '.format(video_image_save_path,exp_name+'_'+image_folder_name+'_fromVideo',exp_name+'_'+image_folder_name+'_fromVideo',exp_name+'_'+image_folder_name+'_fromVideo',save_image_format))


    # os.system('rm -r {}'.format(video_image_save_path))




if __name__ == '__main__':
    videos_path = '/disk4/zexin/project/mice/datasets/0604/'
    vs = os.listdir(videos_path)
    for v in vs:
        runOnOneVideo(videos_path,v)