import numpy as np
import os
import json
import copy
import heapq
from munkres import Munkres, print_matrix
from PIL import Image
# import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import *
# from matching import orb_matching
import argparse
import cv2
from functools import cmp_to_key
import time


def toFixNum_notrack(track_in,frame_list, max_pid_id_setting):
    for idx, frame_name in enumerate(tqdm(frame_list)):
        if len(track_in[frame_name])<=max_pid_id_setting:
            continue
        ## make the first max_pid_id_setting persons  the person with max pose score.
        current_select_score = []
        current_select_score_pid_dict = {}
        for pid in range(max_pid_id_setting):
            current_select_score.append(track_in[frame_name][pid]['scores'])
            current_select_score_pid_dict[track_in[frame_name][pid]['scores']] = pid

        for pid in range(max_pid_id_setting, len(track_in[frame_name])):
            current_min_score = min(current_select_score)
            if track_in[frame_name][pid]['scores'] > min(current_select_score):
                min_score_pid = current_select_score_pid_dict[min(current_select_score)]
                track_in[frame_name][min_score_pid] = track_in[frame_name][pid]
                current_select_score_pid_dict[track_in[frame_name][pid]['scores']] = min_score_pid
                current_select_score.remove(min(current_select_score))
                current_select_score.append(track_in[frame_name][pid]['scores'])
                # track_in[frame_name][pid] = None

        track_in[frame_name] = track_in[frame_name][:max_pid_id_setting]

        # track_in[frame_name]['num_boxes'] = max_pid_id_setting

    return track_in


from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import copy

def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]]) # location and velocity
    kf.F = np.array([[1., dt],
                     [0.,  1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])    # Measurement function
    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix 
    else:
        kf.P[:] = P               # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf

def init_kfs(track_forJson,max_pid_id_setting,num_pose,init_id):
    all_keys = list(track_forJson.keys())
    all_keys.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))
    
    kfs =  [ [] for i in range(max_pid_id_setting)]
    fame_0_key = all_keys[init_id]
    fame_1_key = all_keys[init_id+1]
    for mouse_i in range(max_pid_id_setting):
        for pose_j in range(num_pose):
            for xOrY in range(2):
                pos_1 = track_forJson[fame_1_key][mouse_i]['keypoints'][3*pose_j+xOrY]
                pos_0 = track_forJson[fame_0_key][mouse_i]['keypoints'][3*pose_j+xOrY]
                velocity = pos_1 - pos_0
                confidence = 0.5*(track_forJson[fame_1_key][mouse_i]['keypoints'][3*pose_j+2]+track_forJson[fame_1_key][mouse_i]['keypoints'][3*pose_j+2])
                kfs[mouse_i].append(
                    pos_vel_filter(x=[10000000,velocity],\
#                                    P=np.diag([((1-confidence)*(abs(pos_1-pos_0)))**2, ((1-confidence)*abs(velocity))**2]),\
                                   P=np.diag([500,5000]),\
                                   ## elem in P is var of position and var of velocity
                                   R = 10*(1-confidence), \
                                   ## R is the measurement uncertainty, elem is var of the uncertainty
                                   Q=0.1, \
                                   ## Q is the process covariance. white nose would be fine
                                   dt=1\
                                   ## time interval of two kalman update
                ))
    return kfs
            
def kalman_get(kfs,track_forJson,frame_id):
    frame_info_list_pred = []
    frame_info_list = track_forJson[frame_id]
    for mouse_i, mouse_info_dict in enumerate(frame_info_list):
        frame_info_list_pred.append(copy.deepcopy(mouse_info_dict))
        if mouse_info_dict:
            for pose_j in range(int(len(mouse_info_dict['keypoints'])/3)):
                for xOrY in range(2):
                    kf = kfs[mouse_i][pose_j*2+xOrY]
#                     print(kf.x)
                    z = mouse_info_dict['keypoints'][pose_j*3+xOrY]
                    confidence = mouse_info_dict['keypoints'][pose_j*3+2]
                    kf.predict()
                    if confidence == -1:
                        kf.update(z,R = 0.00000001)
                    elif confidence < 0.15:
                        kf.update(z,R = 100*(1-confidence))
                    elif confidence < 0.25:
                        kf.update(z,R = 50*(1-confidence))
                    else:
                        kf.update(z,R = 0.01*(1-confidence))
                        
#                     kf.update(z)
                    
                    frame_info_list_pred[mouse_i]['keypoints'][pose_j*3+xOrY] = kf.x[0]
    return frame_info_list_pred

def fill_blank_with_predict(track_forJson,frame_id,sorted_keys):
    frame_key = sorted_keys[frame_id]
    frame_info_list = track_forJson[frame_key]
    for mouse_i in range(len(frame_info_list)):
        mouse_info_dict = frame_info_list[mouse_i]
        if not mouse_info_dict and frame_id>3:
            mouse0_info_dict = track_forJson[sorted_keys[frame_id-2]][mouse_i]
            mouse1_info_dict = track_forJson[sorted_keys[frame_id-1]][mouse_i]
            if not mouse0_info_dict or not mouse1_info_dict:
                continue
            mouse_info_dict['idx'] = mouse_i + 1
            mouse_info_dict['scores'] = (mouse0_info_dict['scores'] + mouse1_info_dict['scores'])/2
            mouse_info_dict['keypoints'] = []
            mouse_info_dict['box'] = [999999,99999,-1,-1] 
            for pose_j in range(int(len(mouse1_info_dict['keypoints'])/3)):
                pred_x = 2*mouse1_info_dict['keypoints'][3*pose_j+0] - mouse0_info_dict['keypoints'][3*pose_j+0]
                pred_y = 2*mouse1_info_dict['keypoints'][3*pose_j+1] - mouse0_info_dict['keypoints'][3*pose_j+1]
                pred_s =  (mouse1_info_dict['keypoints'][3*pose_j+2] + mouse0_info_dict['keypoints'][3*pose_j+2])/2
                mouse_info_dict['keypoints'] += [pred_x,pred_y,pred_s]
                mouse_info_dict['box'][0] = min(pred_x, mouse_info_dict['box'][0])
                mouse_info_dict['box'][1] = min(pred_y, mouse_info_dict['box'][1])
                mouse_info_dict['box'][2] = max(pred_x, mouse_info_dict['box'][2])
                mouse_info_dict['box'][3] = max(pred_y, mouse_info_dict['box'][3])
            track_forJson[frame_key][mouse_i] = mouse_info_dict
    return track_forJson
            
def select_slowPose(frame_info_list_pred,track_forJson,frame_id,sorted_keys):
    frame_key = sorted_keys[frame_id]
    frame_info_list = track_forJson[frame_key]
    for mouse_i in range(len(frame_info_list)):
        mouse0_info_dict = track_forJson[sorted_keys[frame_id-1]][mouse_i]
        mouse_info_dict = frame_info_list[mouse_i]
        mouse_info_dict_pred = frame_info_list_pred[mouse_i]
        if not mouse0_info_dict or not mouse_info_dict or not mouse_info_dict_pred:
            continue
        for pose_j in range(int(len(mouse0_info_dict['keypoints'])/3)):
            for xOrY in range(2):
                v_pred = abs(mouse_info_dict_pred['keypoints'][3*pose_j+xOrY] - mouse0_info_dict['keypoints'][3*pose_j+xOrY])
                v = abs(mouse_info_dict['keypoints'][3*pose_j+xOrY] - mouse0_info_dict['keypoints'][3*pose_j+xOrY])
                if (v_pred> 2*v):
                    frame_info_list_pred[mouse_i]['keypoints'][3*pose_j+xOrY] = mouse_info_dict['keypoints'][3*pose_j+xOrY]
    return frame_info_list_pred
    
    
def post_process_tracking(track_forJson,args):
    try:
        frame_start = args.frame_start
        frame_end = args.frame_end
    except Exception as e:
        frame_start = 0
        frame_end = len(sorted_keys)
    if args.kalman:
        kfs = init_kfs(track_forJson,args.max_pid_id_setting,args.num_pose,init_id=frame_start+1)  
    if args.fill_blank_with_predict and args.max_pid_id_setting==-1:
        print('Warning! If fill_blank_with_predict==True, then max_pid_id_setting would better not be -1.')
    
    sorted_keys = list(track_forJson.keys())
    sorted_keys.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))
    
    for frame_id in tqdm(range(frame_start,frame_end)):
#     for frame_id in range(155,170):
        imgname = sorted_keys[frame_id]
        if args.kalman:
            frame_info_list_pred = kalman_get(kfs,track_forJson,imgname)
        else:
            frame_info_list_pred = track_forJson[imgname] 
        if args.smooth_pose:
            frame_info_list_pred = select_slowPose(frame_info_list_pred,track_forJson,frame_id,sorted_keys)
       
        if args.kalman and frame_id>10:
            track_forJson[imgname] = frame_info_list_pred
        
        if args.fill_blank_with_predict:
            track_forJson = fill_blank_with_predict(track_forJson,frame_id,sorted_keys)
            
            
#         ### draw 
#         pairs = [[0,1],[0,2],[0,3]]
#         joint_thred = 0.0
#         img = cv2.imread(os.path.join(args.imgdir,args.image_format%(imgname)))
#         width, height = img.shape[1],img.shape[0]
#         cv2.putText(img, text=imgname,  org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=3)
#         for pid in range(len(track_forJson[imgname])):
#             pose = np.array(track_forJson[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
#             for idx_c in range(pose.shape[0]):
#                 if(pose[idx_c,2]<joint_thred):
#                     continue
#                 cv2.circle(img,center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),radius=6,\
#                            color=((50*idx_c)%255,(80*idx_c)%255,(120*idx_c)%255),thickness=-1)
#                 cv2.putText(img, text='%0.2f'%(pose[idx_c,2]), org=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, \
#                             color=(255,0,0), thickness=2)

#             for idx in range(len(pairs)):
#                 if(pose[pairs[idx][0],2]<joint_thred or pose[pairs[idx][1],2]<joint_thred):
#                     continue
#                 cv2.line(img, pt1=(int(np.clip(pose[pairs[idx][0],0],0,width)),int(np.clip(pose[pairs[idx][0],1],0,height))), pt2=(int(np.clip(pose[pairs[idx][1],0],0,width)),int(np.clip(pose[pairs[idx][1],1],0,height))),\
#                          #color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), thickness=3)
#                          color=(255,0,255), thickness=3)
#         visImage(img)


def display_track_forJson(track_forJson,args):
    img_tmp = cv2.imread(os.path.join(args.imgdir,args.image_format%(list(track_forJson.keys())[0])))
    height, width, channels = img_tmp.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(args.out_video_path, fourcc, 20.0, (width, height))
    
    sorted_keys = list(track_forJson.keys())
    sorted_keys.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))
    for frame_id in tqdm(range(len(sorted_keys))):
    # for frame_id in range(155,170):
        imgname = sorted_keys[frame_id]
        ### draw 
        pairs = [[0,1],[0,2],[0,3]]
        joint_thred = 0.0
        img = cv2.imread(os.path.join(args.imgdir,args.image_format%(imgname)))
        width, height = img.shape[1],img.shape[0]
        cv2.putText(img, text=imgname,  org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=3)
        for pid in range(len(track_forJson[imgname])):
            if not track_forJson[imgname][pid]:
                continue
            pose = np.array(track_forJson[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
            for idx_c in range(pose.shape[0]):
                if(pose[idx_c,2]<joint_thred):
                    continue
                cv2.circle(img,center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),radius=6,\
                           color=((50*idx_c)%255,(80*idx_c)%255,(120*idx_c)%255),thickness=-1)
                cv2.putText(img, text='%0.2f'%(pose[idx_c,2]), org=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, \
                            color=(255,0,0), thickness=2)

            for idx in range(len(pairs)):
                if(pose[pairs[idx][0],2]<joint_thred or pose[pairs[idx][1],2]<joint_thred):
                    continue
                cv2.line(img, pt1=(int(np.clip(pose[pairs[idx][0],0],0,width)),int(np.clip(pose[pairs[idx][0],1],0,height))), pt2=(int(np.clip(pose[pairs[idx][1],0],0,width)),int(np.clip(pose[pairs[idx][1],1],0,height))),\
                         color=((160*pid)%255,(80*pid)%255,(30*pid)%255), thickness=3)
                         # color=(255,0,255), thickness=3)
        out.write(img) # Write out frame to video
    out.release()

def display_pose_cv2(imgdir, visdir, track_forJson,kfs, cmap, args):

    print("Start visualization...\n")
    colors =['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
    part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    pairs = [[0,1],[0,2],[0,3]]
    
    img_tmp = cv2.imread(os.path.join(imgdir,args.image_format%(list(track_forJson.keys())[0])))
    height, width, channels = img_tmp.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(args.out_video_path, fourcc, 20.0, (width, height))
   
    aa = list(track_forJson.keys())
    aa.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))
    # for imgname in tqdm(aa[141:10180]):
    for imgname in tqdm(aa):
        img = cv2.imread(os.path.join(imgdir,args.image_format%(imgname)))
        width, height = img.shape[1],img.shape[0]
        cv2.putText(img, text=imgname,  org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=3)
        
        
        frame_info_list_pred = kalman_get(kfs,track_forJson,imgname)

        for pid in range(len(track_forJson[imgname])):
            if not track_forJson[imgname][pid]:
                continue
            # print(track_forJson[imgname][pid]['scores'])
            pose = np.array(track_forJson[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
            joint_thred = 0.0
            tracked_id = track_forJson[imgname][pid]['idx']
            cv2.putText(img,  text=str(tracked_id), org=(int(np.clip(pose[0,0],0,width)), int(np.clip(pose[0,1],0,height))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=int(1*np.mean(pose[:,2])+1), color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), thickness=3)
            
            # ## draw output of the neuron network
            # for idx_c in range(pose.shape[0]):
            #     if(pose[idx_c,2]<joint_thred):
            #         continue
            #     cv2.circle(img,center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),radius=6,\
            #                color=((50*idx_c)%255,(80*idx_c)%255,(120*idx_c)%255),thickness=-1)
            #     cv2.putText(img, text='%0.2f'%(pose[idx_c,2]), org=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, \
            #                 color=(255,0,0), thickness=2)
            #     # print('  ',pose[idx_c,2])
            
            # for idx in range(len(pairs)):
            #     if(pose[pairs[idx][0],2]<joint_thred or pose[pairs[idx][1],2]<joint_thred):
            #         continue
            #     cv2.line(img, pt1=(int(np.clip(pose[pairs[idx][0],0],0,width)),int(np.clip(pose[pairs[idx][0],1],0,height))), pt2=(int(np.clip(pose[pairs[idx][1],0],0,width)),int(np.clip(pose[pairs[idx][1],1],0,height))),\
            #              # color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), thickness=3)
            #              color=(255,0,255), thickness=3)

            ## draw output of the kalman filter
            pose = np.array(frame_info_list_pred[pid]['keypoints']).reshape(-1,3)[:,:3]
            for idx_c in range(pose.shape[0]):
                if(pose[idx_c,2]<joint_thred):
                    continue
                cv2.circle(img,center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),radius=6,color=(0,(80*idx_c)%255,(120*idx_c)%255),thickness=-1)
            for idx in range(len(pairs)):
                if(pose[pairs[idx][0],2]<joint_thred or pose[pairs[idx][1],2]<joint_thred):
                    continue
                cv2.line(img, pt1=(int(np.clip(pose[pairs[idx][0],0],0,width)),int(np.clip(pose[pairs[idx][0],1],0,height))), pt2=(int(np.clip(pose[pairs[idx][1],0],0,width)),int(np.clip(pose[pairs[idx][1],1],0,height))), \
                         # color=(0,(80*tracked_id)%255,(30*tracked_id)%255), thickness=3)
                         color=(255,255,0), thickness=3)


        if not os.path.exists(visdir): 
            os.mkdir(visdir)
        # visImage(img)
        out.write(img) # Write out frame to video
    out.release()
    visImage(img)
    print('demo image is generated in ',visdir)
    print('demo video is generated as: ',args.out_video_path)