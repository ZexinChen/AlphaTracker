# -*- coding:UTF-8 -*-
######## import for python2  ########
from __future__ import print_function
######## import for python2 ########
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pandas as pd
import os
import cv2
from PIL import Image
import math
import pickle
from functools import cmp_to_key
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from tqdm import tqdm
import json
import re
import time 
from sklearn import metrics
import csv
import umap
import random


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if os.path.exists(path) and os.path.isdir(path):
            pass
        else: raise

# def RGB_to_Hex(tmp):
def RGB_to_Hex(rgb):
    # rgb = tmp.split(',')
    strs = '#'
    for num in rgb:
        # num = int(i)#str to int
        #RGB to hex and upcap
        strs += str(hex(num))[-2:].replace('x','0').upper()
        
    return strs

def square_sum(vec1,vec2):
    all_sum = 0
    for v1,v2 in zip(vec1,vec2):
        all_sum += (v1-v2)**2
    return all_sum

def interval_in(start,end, intervals):
    if len(intervals)==0:
        return True
    else:
        for itv in intervals:
            if start>=itv[0] and end<=itv[1]:
                return True
    return False

def select_clips(folder_dict,intervals,arg):
    files = os.listdir(folder_dict)

    frames_0 = []
    frames_1 = []
    for file in files:
        fs = file.split('_')
        if fs[1] == 'mouse0':
            frames_0.append(int(fs[-1][0:-4]))
        else:
            frames_1.append(int(fs[-1][0:-4]))

    frames_0 = sorted(frames_0)
    frames_1 = sorted(frames_1)

    frame_share = []
    fid_0,fid_1 = 0,0
    while(fid_0<len(frames_0) and fid_1<len(frames_1)):
        if frames_0[fid_0] == frames_1[fid_1]:
            frame_share.append(frames_0[fid_0])
            fid_0+=1
            fid_1+=1
        elif frames_0[fid_0] < frames_1[fid_1]:
            fid_0+=1
        else:
            fid_1+=1
    frame_inds = []
    frames = frame_share
    while len(frames)>= arg.clip_length:
        flag = frames[0]
        if frames[arg.clip_length-1] == flag + arg.clip_length -1 and interval_in(start=frames[0],end=frames[1], intervals=intervals):
            frame_inds.append(frames[0:arg.clip_length])
            # print(frame_inds)
            del frames[0:arg.clip_length]
        else:
            del frames[0]

    return frame_inds


def retrieve_poses(arg):

    import json

    json_names = arg.tracked_json

    with open('./utils_file/pca.pckl','rb') as f:
        faces_pca = pickle.load(f)

    pose_clips = []
    info_clips = []
    cont_clips = []
    # for i in range(len(json_names)):
    for i in [1]:

        print('%d/%d retrieve poses from %s '%(i+1,len(json_names), json_names[i]))
        with open(json_names[i],'r') as json_file:
            data = json.loads(json_file.read())

        data_keys = list(data.keys())
        data_keys.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))

        frame_inds = select_clips(arg.contdir[i],arg.intervals[i])
        frame_inds = np.asarray(frame_inds)
        # clip_num,_ = frame_inds.shape #python2
        clip_num = frame_inds.shape[0] #python3

        for clip_ind in tqdm(range(clip_num)):

            pose_clip = []
            info_clip = []
            cont_clip = []
            j = 0
            for frame_ind in frame_inds[clip_ind,:]:
                if(frame_ind<0):
                    print(frame_inds[clip_ind,:])

                mice = data[data_keys[frame_ind]]
                try:
                    # print(i,clip_ind,j,json_names[i])
                    if mice[0]['idx']==1:
                        pose = np.asarray(mice[0]['keypoints'])
                    else:
                        pose = np.asarray(mice[1]['keypoints'])
                except Exception as e:
                    print('failed to get pose:')
                    print(json_names[i])
                    raise e
                
                pose = pose.reshape((4,3))
                pose = np.delete(pose, 2, axis=1)
                pose_clip.append(pose[:])
                info_clip.append(arg.imgdir[i] + data_keys[frame_ind] + '.png')

                faces = pd.DataFrame([])
                frame = plt.imread(arg.contdir[i] + 'mask_{}.png'.format(frame_ind))
                frame = np.asarray(frame[:,:,0],dtype = 'uint8')
                face = pd.Series(frame.flatten(),name = frame_ind)
                faces = faces.append(face)

                component = faces_pca.transform(faces)

                cont_clip.append(component[0][0:10])

                j = j+1

            pose_clips.append(pose_clip)
            info_clips.append(info_clip)
            cont_clips.append(cont_clip)


    return arg, pose_clips, info_clips, cont_clips

def distBetweenTwoMice(p1,p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def retrieve_poses_Mice(arg):

    json_names = arg.tracked_json

    with open('./utils_file/pca.pckl','rb') as f:
        faces_pca = pickle.load(f)

    pose_clips = []
    poseTheOther_clips = []
    info_clips = []
    frames_path_clips = []
    cont_clips = []
    contTheOther_clips = []
    number_of_clip_used = 0
    for i in range(min(arg.max_clips_num*3,len(json_names))):

        if len(pose_clips)>arg.max_clips_num*6:
            print("max_clips_num is set to be {}, {} clips are retrieved. There should be enough clip left after cleaning data process. if not, please increase max_clips_num".format(arg.max_clips_num,arg.max_clips_num*6))
            break


        print('%d/%d retrieve poses from %s '%(i+1,len(json_names), json_names[i]))
        with open(json_names[i],'r') as json_file:
            data = json.loads(json_file.read())

        data_keys = list(data.keys())    
        data_keys.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))
        

        clip_id = 0
        frame_id = 0
        pose_clip = []
        poseTheOther_clip = []
        cont_clip = []
        contTheOther_clip = []
        frames_path_clip = []
        info_dict = {'track_infos':[],'frame_keys':[],'video_path':arg.imgdir[i],'clip_id':'%d_%d'%(i,clip_id)}
        frame_ind_inClip = 0
        while frame_id < len(data_keys):
            print('\rprocessing frame No.{} into clip No.{} as the {} frame'.format(frame_id, clip_id, frame_ind_inClip),end='')
            if frame_ind_inClip == arg.clip_length:
                # all info is collected for one clip
                clip_id += 1
                pose_clips.append(pose_clip)
                poseTheOther_clips.append(poseTheOther_clip)
                cont_clips.append(cont_clip)
                contTheOther_clips.append(contTheOther_clip)
                frames_path_clips.append(frames_path_clip)
                info_clips.append(info_dict)
                pose_clip = []
                poseTheOther_clip = []
                cont_clip = []
                contTheOther_clip = []
                frames_path_clip = []
                info_dict = {'track_infos':[],'frame_keys':[],'video_path':arg.imgdir[i],'clip_id':'%d_%d'%(i,clip_id)}
                frame_ind_inClip = 0
                if len(pose_clips)>arg.max_clips_num*6:
                    print("max_clips_num is set to be {}, {} clips are retrieved. There should be enough clip left after cleaning data process. if not, please increase max_clips_num".format(arg.max_clips_num,arg.max_clips_num*6))
                    break

            if (arg.json_keyFormat == ''):
                data_key = data_keys[frame_id]
            else:
                data_key = arg.json_keyFormat%(frame_id)
            if data_key not in data_keys:
                print('can not find key {} in json. (If all the frame have this error. Please check json_keyFormat in setting.py)'.format(data_key))
                frame_id += 1
                pose_clip = []
                poseTheOther_clip = []
                cont_clip = []
                contTheOther_clip = []
                frames_path_clip = []
                info_dict = {'track_infos':[],'frame_keys':[],'video_path':arg.imgdir[i],'clip_id':'%d_%d'%(i,clip_id)}
                frame_ind_inClip = 0
                continue

            try:
                mice = data[data_key]
                miceCenters = [np.asarray(mice[m_id]['keypoints']).reshape((arg.joint_num,3)).mean(axis=0) for m_id in range(len(mice))]
                dist = [distBetweenTwoMice(miceCenters[arg.targetMouseID],miceCenters[i]) for i in range(len(miceCenters))]
                dist[arg.targetMouseID] = max(dist) + 1
                referMouseID = dist.index(min(dist))
                pose = np.asarray(mice[arg.targetMouseID]['keypoints'])
                poseTheOther = np.asarray(mice[referMouseID]['keypoints'])
                    
                pose = pose.reshape((arg.joint_num,3))
                pose = np.delete(pose, 2, axis=1)
                pose_clip.append(pose[:])

                poseTheOther = poseTheOther.reshape((arg.joint_num,3))
                poseTheOther = np.delete(poseTheOther, 2, axis=1)
                poseTheOther_clip.append(poseTheOther[:])

                frames_path_clip.append(arg.imgdir[i] + data_key + '.png')
                info_dict['track_infos'].append(mice)
                info_dict['frame_keys'].append(data_key)

                faces = pd.DataFrame([])
                frame = plt.imread(arg.contdir[i] + 'mask_mouse{}_{}.png'.format(arg.targetMouseID,frame_id))
                frame = np.asarray(frame[:,:,0],dtype = 'uint8')
                face = pd.Series(frame.flatten(),name = frame_id)
                faces = faces.append(face)
                component = faces_pca.transform(faces)
                cont_clip.append(component[0][0:10])

                faces = pd.DataFrame([])
                frame = plt.imread(arg.contdir[i] + 'mask_mouse{}_{}.png'.format(referMouseID,frame_id))
                frame = np.asarray(frame[:,:,0],dtype = 'uint8')
                face = pd.Series(frame.flatten(),name = frame_id)
                faces = faces.append(face)
                component = faces_pca.transform(faces)
                contTheOther_clip.append(component[0][0:10])

                frame_id += 1
                frame_ind_inClip += 1

            except Exception as e:
                print('failed to process the {}th frame with error:{}. will restart the clip'.format(frame_id,e))
                frame_id += 1
                pose_clip = []
                poseTheOther_clip = []
                cont_clip = []
                contTheOther_clip = []
                frames_path_clip = []
                info_dict = {'track_infos':[],'frame_keys':[],'video_path':arg.imgdir[i],'clip_id':'%d_%d'%(i,clip_id)}
                frame_ind_inClip = 0
                
    pose_clips = np.asarray(pose_clips)
    poseTheOther_clips = np.asarray(poseTheOther_clips)
    cont_clips = np.asarray(cont_clips)
    contTheOther_clips = np.asarray(contTheOther_clips)

    return arg, {'pose_clips':pose_clips, 
        'poseTheOther_clips':poseTheOther_clips, 
        'cont_clips':cont_clips, 
        'contTheOther_clips':contTheOther_clips, 
        'frames_path_clips':frames_path_clips,
        'info_clips':info_clips}



def retrieve_poses_twoMice(arg):

    json_names = arg.tracked_json

    with open('./utils_file/pca.pckl','rb') as f:
        faces_pca = pickle.load(f)

    pose_clips = []
    poseTheOther_clips = []
    info_clips = []
    frames_path_clips = []
    cont_clips = []
    contTheOther_clips = []
    for i in range(len(json_names)):

        print('%d/%d retrieve poses from %s '%(i+1,len(json_names), json_names[i]))
        with open(json_names[i],'r') as json_file:
            data = json.loads(json_file.read())

        data_keys = list(data.keys())
        data_keys.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))

        frame_inds = select_clips(arg.contdir[i],arg.intervals[i],arg)
        frame_inds = np.asarray(frame_inds)
        # clip_num,_ = frame_inds.shape #python2
        clip_num = frame_inds.shape[0] #python3

        for clip_ind in tqdm(range(clip_num)):
        # for clip_ind in tqdm(range(0,min(clip_num,10))):

            pose_clip = []
            poseTheOther_clip = []
            cont_clip = []
            contTheOther_clip = []
            frames_path_clip = []
            info_dict = {'track_infos':[],'frame_keys':[],'video_path':arg.imgdir[i],'clip_id':'%d_%d'%(i,clip_ind)}
            j = 0
            for frame_ind in frame_inds[clip_ind,:]:
                if(frame_ind<0):
                    print(frame_inds[clip_ind,:])

                if (arg.json_keyFormat == ''):
                    data_key = data_keys[frame_ind]
                else:
                    data_key = arg.json_keyFormat%(frame_ind)
                if data_key not in data_keys:
                    break

                mice = data[data_key]
                try:
                    # print(i,clip_ind,j,json_names[i])
                    if arg.targetMouseID == 0:
                        pose = np.asarray(mice[0]['keypoints'])
                        poseTheOther = np.asarray(mice[1]['keypoints'])
                        referMouseID = 1
                    else:
                        pose = np.asarray(mice[1]['keypoints'])
                        poseTheOther = np.asarray(mice[0]['keypoints'])
                        referMouseID = 0
                except Exception as e:
                    if(len(mice)<=1 or 'keypoints' not in mice[0] or 'keypoints' not in mice[1]):
                    # if(len(mice)<=1):
                        j = j+1
                        continue
                    print('failed to get pose of %s'%(json_names[i]))
                    print(mice)
                    # j = j+1
                    # continue
                    raise e
                
                pose = pose.reshape((arg.joint_num,3))
                pose = np.delete(pose, 2, axis=1)
                pose_clip.append(pose[:])

                poseTheOther = poseTheOther.reshape((arg.joint_num,3))
                poseTheOther = np.delete(poseTheOther, 2, axis=1)
                poseTheOther_clip.append(poseTheOther[:])

                frames_path_clip.append(arg.imgdir[i] + data_key + '.png')
                info_dict['track_infos'].append(mice)
                info_dict['frame_keys'].append(data_key)

                faces = pd.DataFrame([])
                frame = plt.imread(arg.contdir[i] + 'mask_mouse{}_{}.png'.format(arg.targetMouseID,frame_ind))
                frame = np.asarray(frame[:,:,0],dtype = 'uint8')
                face = pd.Series(frame.flatten(),name = frame_ind)
                faces = faces.append(face)
                component = faces_pca.transform(faces)
                cont_clip.append(component[0][0:10])

                faces = pd.DataFrame([])
                frame = plt.imread(arg.contdir[i] + 'mask_mouse{}_{}.png'.format(referMouseID,frame_ind))
                frame = np.asarray(frame[:,:,0],dtype = 'uint8')
                face = pd.Series(frame.flatten(),name = frame_ind)
                faces = faces.append(face)
                component = faces_pca.transform(faces)
                contTheOther_clip.append(component[0][0:10])


                j = j+1

            pose_clips.append(pose_clip)
            poseTheOther_clips.append(poseTheOther_clip)
            info_clips.append(info_dict)
            frames_path_clips.append(frames_path_clip)
            cont_clips.append(cont_clip)
            contTheOther_clips.append(contTheOther_clip)

    pose_clips = np.asarray(pose_clips)
    poseTheOther_clips = np.asarray(poseTheOther_clips)
    cont_clips = np.asarray(cont_clips)
    contTheOther_clips = np.asarray(contTheOther_clips)


    return arg, {'pose_clips':pose_clips, 
        'poseTheOther_clips':poseTheOther_clips, 
        'cont_clips':cont_clips, 
        'contTheOther_clips':contTheOther_clips, 
        'frames_path_clips':frames_path_clips,
        'info_clips':info_clips}


def clean_differentLength_clips(clips_dict):
    pose_clips_filtered, poseTheOther_clips_filtered, info_clips_filtered, cont_clips_filtered, contTheOther_clips_filtered,frames_path_clips_filtered = [],[],[],[],[],[]
    pose_clips = clips_dict['pose_clips']
    if len(pose_clips) == 0:
        print('there are no clip to process. Please check if the script in ./utils_file run currectly and the json is correct')
        raise
    clip_shape = np.asarray(pose_clips[0]).shape
    delete_clip_num = 0
    for c_id in range(len(pose_clips)):
        if np.asarray(pose_clips[c_id]).shape == clip_shape:
            pose_clips_filtered.append(clips_dict['pose_clips'][c_id])
            poseTheOther_clips_filtered.append(clips_dict['poseTheOther_clips'][c_id])
            cont_clips_filtered.append(clips_dict['cont_clips'][c_id])
            contTheOther_clips_filtered.append(clips_dict['contTheOther_clips'][c_id])
            info_clips_filtered.append(clips_dict['info_clips'][c_id])
            frames_path_clips_filtered.append(clips_dict['frames_path_clips'][c_id])
        else:
            delete_clip_num+=1
    print('There are {} clips which dont have enough frames'.format(delete_clip_num))

    pose_clips_filtered = np.asarray(pose_clips_filtered)
    poseTheOther_clips_filtered = np.asarray(poseTheOther_clips_filtered)
    cont_clips_filtered = np.asarray(cont_clips_filtered)
    contTheOther_clips_filtered = np.asarray(contTheOther_clips_filtered)

    return {'pose_clips':pose_clips_filtered, 
        'poseTheOther_clips':poseTheOther_clips_filtered, 
        'cont_clips':cont_clips_filtered, 
        'contTheOther_clips':contTheOther_clips_filtered, 
        'frames_path_clips':frames_path_clips_filtered,
        'info_clips':info_clips_filtered}

def remove_longMiceDist_clips(cfg,clips_dict):
    pose_clips_filtered, poseTheOther_clips_filtered, info_clips_filtered, cont_clips_filtered, contTheOther_clips_filtered,frames_path_clips_filtered = [],[],[],[],[],[]
    pose_clips = clips_dict['pose_clips']
    poseTheOther_clips = clips_dict['poseTheOther_clips']
    numOfClipTooFar = 0
    for clip_ind in range(len(pose_clips)):
        nose_xy_Nx2 = pose_clips[clip_ind,:,0,:]
        noseTO_xy_Nx2 = poseTheOther_clips[clip_ind,:,0,:]
        distance = np.min(np.linalg.norm(nose_xy_Nx2-noseTO_xy_Nx2))
        bodyVector_xy_Nx2 = pose_clips[clip_ind,:,0,:] - pose_clips[clip_ind,:,3,:]
        if distance < np.max(np.linalg.norm(bodyVector_xy_Nx2))* cfg.distance_threshold:
            pose_clips_filtered.append(clips_dict['pose_clips'][clip_ind])
            poseTheOther_clips_filtered.append(clips_dict['poseTheOther_clips'][clip_ind])
            cont_clips_filtered.append(clips_dict['cont_clips'][clip_ind])
            contTheOther_clips_filtered.append(clips_dict['contTheOther_clips'][clip_ind])
            info_clips_filtered.append(clips_dict['info_clips'][clip_ind])
            frames_path_clips_filtered.append(clips_dict['frames_path_clips'][clip_ind])
        else:
            numOfClipTooFar += 1
    print('There are {} clips where mice are too far from each other. Removed'.format(numOfClipTooFar))


    pose_clips_filtered = np.asarray(pose_clips_filtered)
    poseTheOther_clips_filtered = np.asarray(poseTheOther_clips_filtered)
    cont_clips_filtered = np.asarray(cont_clips_filtered)
    contTheOther_clips_filtered = np.asarray(contTheOther_clips_filtered)

    return {'pose_clips':pose_clips_filtered, 
        'poseTheOther_clips':poseTheOther_clips_filtered, 
        'cont_clips':cont_clips_filtered, 
        'contTheOther_clips':contTheOther_clips_filtered, 
        'frames_path_clips':frames_path_clips_filtered,
        'info_clips':info_clips_filtered} 


def align_poses_self(pose_clips):

    norm_pose_clips = pose_clips.copy()
    clip_num,frame_num,point_num,_ = norm_pose_clips.shape
    # mid_frame_ind = 7
    mid_frame_ind = int(frame_num/2)

    for clip_ind in range(clip_num):
        pose_clip = norm_pose_clips[clip_ind,:,:,:]

        pose_clip[:,:,0] -= pose_clip[mid_frame_ind,0,0]
        pose_clip[:,:,1] -= pose_clip[mid_frame_ind,0,1]

        # fig,ax = plt.subplots()
        # for ii in range(4):
        #     ax.plot(pose_clip[7,ii,0],pose_clip[7,ii,1],'bo')

        mid_pose = pose_clip[mid_frame_ind,:,:]
        mid_nose = mid_pose[0,:]
        mid_tail = mid_pose[3,:]
        mid_body = mid_tail-mid_nose

        rho, phi = cart2pol(mid_body[0], mid_body[1])
        # print(math.degrees(phi))
        c, s = np.cos(phi), np.sin(phi)
        j = np.matrix([[c, s], [-s, c]])

        for ii in range(frame_num):
            for jj in range(point_num):
                pose_clip[ii,jj,:] = np.dot(j,pose_clip[ii,jj,:])

        norm_pose_clips[clip_ind,:,:,:] = pose_clip

    return norm_pose_clips

def align_poses_toTheFirst(pose_clips, poseTheOther_clips):

    norm_poseTheOther_clips = poseTheOther_clips.copy()
    clip_num,frame_num,point_num,_ = norm_poseTheOther_clips.shape
    # mid_frame_ind = 7
    mid_frame_ind = int(frame_num/2)

    for clip_ind in range(clip_num):
        pose_clip = pose_clips[clip_ind,:,:,:]
        poseTheOther_clip = norm_poseTheOther_clips[clip_ind,:,:,:]

        ### align position ###
        poseTheOther_clip[:,:,0] -= pose_clip[mid_frame_ind,0,0]
        poseTheOther_clip[:,:,1] -= pose_clip[mid_frame_ind,0,1]

        ### align angle ###
        mid_pose = pose_clip[mid_frame_ind,:,:]
        mid_nose = mid_pose[0,:]
        mid_tail = mid_pose[3,:]
        mid_body = mid_tail-mid_nose
        rho, phi = cart2pol(mid_body[0], mid_body[1])
        c, s = np.cos(phi), np.sin(phi)
        j = np.matrix([[c, s], [-s, c]])
        for ii in range(frame_num):
            for jj in range(point_num):
                poseTheOther_clip[ii,jj,:] = np.dot(j,poseTheOther_clip[ii,jj,:])

        norm_poseTheOther_clips[clip_ind,:,:,:] = poseTheOther_clip

    return norm_poseTheOther_clips

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    div = np.linalg.norm(vector)
    if div == 0 :
        # print(vector)
        return vector
    else:
        return vector / div

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    # print(v1_u)
    v2_u = unit_vector(v2)
    # print(v1_u)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # print('angle',angle)
    _,phi1 = cart2pol(v1[0],v1[1])
    phi1 = math.degrees(phi1)

    _,phi2 = cart2pol(v2[0],v2[1])
    phi2 = math.degrees(phi2)


    if (phi1<phi2):
        if (phi2-phi1)<180:
            flag = True
        else:
            flag = False
    else:
        if (phi1-phi2)<180:
            flag = False
        else:
            flag = True
    if flag:
        angle = - angle

    return angle, np.cos(angle),np.sin(angle)

def compute_features_pos(arg,norm_pose_clips,info_clips,cont_clips):
    clip_num,frame_num,point_num,_ = norm_pose_clips.shape
    feature_clips = []
    raw_feature_clips = []

    for clip_ind in range(clip_num):

        # ## FFT of nose
        # point = 0
        # vec = norm_pose_clips[clip_ind,:,point,0] + 1j * norm_pose_clips[clip_ind,:,point,1]
        # np_fft = np.fft.fft(vec)
        # amplitudes = 2/vec.size * np.abs(np_fft)
        # angles = np.angle(np_fft)

        # feature_clip = np.concatenate((amplitudes[0:8], np.cos(angles)[0:8]))
        # feature_clip = np.concatenate((feature_clip,np.sin(angles)[0:8]))
        # raw_feature_clip = np.concatenate((norm_pose_clips[clip_ind,:,point,0],norm_pose_clips[clip_ind,:,point,1]))

        ### Displacement angle
        # displace_x = norm_pose_clips[clip_ind,:,0,0]
        # displace_y = norm_pose_clips[clip_ind,:,0,1]
        # feature_clip = np.concatenate((feature_clip,displace_x))
        # feature_clip = np.concatenate((feature_clip,displace_y))
        # raw_feature_clip = np.concatenate((raw_feature_clip,displace_x))
        # raw_feature_clip = np.concatenate((raw_feature_clip,displace_y))

        ### Displacement angle only
        displace_x = norm_pose_clips[clip_ind,:,0,0]
        displace_y = norm_pose_clips[clip_ind,:,0,1]
        feature_clip = np.concatenate((displace_y,displace_x))
        raw_feature_clip = np.concatenate((displace_y,displace_x))

        feature_clips.append(feature_clip)
        raw_feature_clips.append(raw_feature_clip)


    return feature_clips,raw_feature_clips,info_clips,cont_clips

def compute_features_sep(arg,norm_pose_clips,info_clips,cont_clips):

    clip_num,frame_num,point_num,_ = norm_pose_clips.shape
    raw_feature_clips = []
    raw_feature_clips_dict = []
    feature_clips_dict = []

    body_length_clips = []
    head_earMid_length_clips = []
    head_body_ang_clips = []
    displacementx_clips = []
    displacementy_clips = []
    displace_rho_clips = []
    displace_phi_s_clips = []
    displace_phi_c_clips = []
    nose_fft_amp_clips = []
    nose_fft_ang_clips = []
    contourPCA_fft_amp_clips = []
    contourPCA_fft_ang_clips = []

    bad_clip = np.zeros((1,clip_num))
    for clip_ind in range(clip_num):
        ### Body Length
        mid_id = arg.clip_length // 2 +1
        mid_pose = norm_pose_clips[clip_ind,mid_id,:,:]
        mid_nose,mid_ear1,mid_ear2,mid_tail = mid_pose[0,0:2], mid_pose[1,0:2], mid_pose[2,0:2], mid_pose[3,0:2]
        mid_body = mid_nose-mid_tail
        if np.linalg.norm(mid_body) == 0:
            bad_clip[clip_ind] = 1
            continue
        mid_head = mid_nose - (mid_ear1+mid_ear2)/2
        if np.linalg.norm(mid_head) == 0:
            bad_clip[clip_ind] = 1
            continue
        body = norm_pose_clips[clip_ind,:,0,:] - norm_pose_clips[clip_ind,:,3,:]
        if np.any(np.linalg.norm(body)==0):
            bad_clip[clip_ind] = 1
            continue
       
        ### Head ear-middle-point vector
        head = norm_pose_clips[clip_ind,:,0,:] - (norm_pose_clips[clip_ind,:,1,:]+norm_pose_clips[clip_ind,:,2,:])/2
        if np.any(np.linalg.norm(head)==0):
            bad_clip[clip_ind] = 1
            continue

        ### Head Body angle
        cos_angles = []
        sin_angles = []
        angles_hb = []
        for i in range(arg.clip_length):
            angle, cos_angle,sin_angle = angle_between(body[i,:],head[i,:])
            if cos_angles==np.nan or sin_angle==np.nan:
                print((body[i,:],head[i,:]))
            cos_angles.append(cos_angle)
            sin_angles.append(sin_angle)
            angles_hb.append(angle)

        ### FFT of nose
        point = 0
        vec = norm_pose_clips[clip_ind,:,point,0] + 1j * norm_pose_clips[clip_ind,:,point,1]
        np_fft = np.fft.fft(vec)
        amplitudes = 2/vec.size * np.abs(np_fft)
        angles = np.angle(np_fft)
        
        ### Displacement
        displace_x = norm_pose_clips[clip_ind,:,0,0]
        displace_y = norm_pose_clips[clip_ind,:,0,1]

        ### Displacement on polar coordinates
        rho_vector = []
        phi_sin_vector = []
        phi_cos_vector = []
        for d_i in range(len(displace_x)):
            d_x = displace_x[d_i]
            d_y = displace_y[d_i]
            rho, phi = cart2pol(d_x, d_y)
            c, s = np.cos(phi), np.sin(phi)
            rho_vector.append(rho)
            phi_sin_vector.append(s)
            phi_cos_vector.append(c)


        ### Contour PCA fft
        np_fft_contour = np.fft.fft(cont_clips[clip_ind,:,:],axis = 0)
        amplitudes_contour = np.abs(np_fft_contour)
        amplitudes_contour = amplitudes_contour[0:8,:]
        amplitudes_contour = amplitudes_contour.flatten()
        angles_contour = np.angle(np_fft_contour)
        angles_contour = angles_contour[0:8,:]
        angles_contour = angles_contour.flatten()


        body_length_clips.append(np.linalg.norm(body, axis = 1))
        head_earMid_length_clips.append(np.linalg.norm(head, axis = 1))
        head_body_ang_clips.append(np.concatenate((np.asarray(sin_angles),np.asarray(cos_angles))))
        displacementx_clips.append(np.asarray(displace_x))
        displacementy_clips.append(np.asarray(displace_y))
        displace_rho_clips.append(np.asarray(rho_vector))
        displace_phi_s_clips.append(np.asarray(phi_sin_vector))
        displace_phi_c_clips.append(np.asarray(phi_cos_vector))
        
        nose_fft_amp_clips.append(np.asarray(amplitudes[0:8]))
        nose_fft_ang_clips.append(np.concatenate((np.sin(angles)[0:8], np.cos(angles)[0:8])))
        contourPCA_fft_amp_clips.append(np.asarray(amplitudes_contour[0:8]))
        contourPCA_fft_ang_clips.append(np.concatenate((np.sin(angles_contour)[0:8], np.cos(angles_contour)[0:8])))
        

        raw_feature_clip_dict={
            'displace_x':displace_x,
            'displace_y':displace_y,
            'displace_rho':rho_vector,
            'displace_phi_s':phi_sin_vector,
            'displace_phi_c':phi_cos_vector,
            'body_length':np.linalg.norm(body, axis = 1),
            'head_length':np.linalg.norm(head, axis = 1),
            'head_body_angles':np.asarray(angles_hb),
        }


        raw_feature_clips_dict.append(raw_feature_clip_dict)

    good_index = np.nonzero(1-bad_clip)
    good_index = list(good_index[1])

    body_length_clips = [body_length_clips[i] for i in good_index]
    head_earMid_length_clips = [head_earMid_length_clips[i] for i in good_index]
    head_body_ang_clips = [head_body_ang_clips[i] for i in good_index]
    nose_fft_amp_clips = [nose_fft_amp_clips[i] for i in good_index]
    nose_fft_ang_clips = [nose_fft_ang_clips[i] for i in good_index]
    displacementx_clips = [displacementx_clips[i] for i in good_index]
    displacementy_clips = [displacementy_clips[i] for i in good_index]
    displace_rho_clips = [displace_rho_clips[i] for i in good_index]
    displace_phi_s_clips = [displace_phi_s_clips[i] for i in good_index]
    displace_phi_c_clips = [displace_phi_c_clips[i] for i in good_index]
    contourPCA_fft_amp_clips = [contourPCA_fft_amp_clips[i] for i in good_index]
    contourPCA_fft_ang_clips = [contourPCA_fft_ang_clips[i] for i in good_index]
    
    info_clips = [info_clips[i] for i in good_index]
    cont_clips = [cont_clips[i] for i in good_index]
    raw_feature_clips_dict = [raw_feature_clips_dict[i] for i in good_index]


    feature_clips_dict={
            'displace_x':displacementx_clips,
            'displace_y':displacementy_clips,
            'displace_rho':displacementx_clips,
            'displace_phi_c':displace_phi_c_clips,
            'displace_phi_s':displace_phi_s_clips,
            'body_length':body_length_clips,
            'head_length':head_earMid_length_clips,
            'head_body_angles':head_body_ang_clips,
            'nose_fft_amp':nose_fft_amp_clips,
            'nose_fft_ang':nose_fft_ang_clips,
            'contourPCA_fft_amp':contourPCA_fft_amp_clips,
            'contourPCA_fft_ang':contourPCA_fft_ang_clips,
            
        }

    return feature_clips_dict,info_clips,cont_clips,raw_feature_clips_dict

def compute_features_sep_twoMice(arg,norm_pose_clips,norm_poseTheOther_clips,info_clips,cont_clips, pose_clips, poseTheOther_clips):

    clip_num,frame_num,point_num,_ = norm_pose_clips.shape
    raw_feature_clips = []
    raw_feature_clips_dict = []
    feature_clips_dict = []

    body_length_clips = []
    head_earMid_length_clips = []
    head_body_ang_clips = []
    displacementx_clips = []
    displacementy_clips = []
    displace_rho_clips = []
    displace_phi_s_clips = []
    displace_phi_c_clips = []
    nose_fft_amp_clips = []
    nose_fft_ang_clips = []
    contourPCA_fft_amp_clips = []
    contourPCA_fft_ang_clips = []

    body_length_clips_TO = []
    head_earMid_length_clips_TO = []
    head_body_ang_clips_TO = []
    displacementx_clips_TO = []
    displacementy_clips_TO = []
    displace_rho_clips_TO = []
    displace_phi_s_clips_TO = []
    displace_phi_c_clips_TO = []
    nose_fft_amp_clips_TO = []
    nose_fft_ang_clips_TO = []
    contourPCA_fft_amp_clips_TO = []
    contourPCA_fft_ang_clips_TO = []

    two_body_ang_clips = []
    nose_tail_displace_rho_clips = []
    nose_tail_displace_phi_clips = []

    info_clips_selected = []
    cont_clips_selected = []
    pose_clips_selected = []
    poseTheOther_clips_selected = []

    # bad_clip = np.zeros((1,clip_num))
    # bad_clip = np.zeros(clip_num)
    good_index = []
    for clip_ind in range(clip_num):
        ### Body Length
        mid_id = arg.clip_length // 2 +1
        mid_pose = norm_pose_clips[clip_ind,mid_id,:,:]
        mid_nose,mid_ear1,mid_ear2,mid_tail = mid_pose[0,0:2], mid_pose[1,0:2], mid_pose[2,0:2], mid_pose[3,0:2]
        mid_body = mid_nose-mid_tail
        mid_head = mid_nose - (mid_ear1+mid_ear2)/2
        body = norm_pose_clips[clip_ind,:,0,:] - norm_pose_clips[clip_ind,:,3,:]
        body_TO = norm_poseTheOther_clips[clip_ind,:,0,:] - norm_poseTheOther_clips[clip_ind,:,3,:]

        if np.linalg.norm(mid_body) == 0 or np.linalg.norm(mid_head) == 0 \
         or np.any(np.linalg.norm(body)==0) or np.any(np.linalg.norm(body_TO)==0):
            print('warning: Body Length == 0')
            continue
       
        ### Head ear-middle-point vector
        head = norm_pose_clips[clip_ind,:,0,:] - (norm_pose_clips[clip_ind,:,1,:]+norm_pose_clips[clip_ind,:,2,:])/2
        head_TO = norm_poseTheOther_clips[clip_ind,:,0,:] - (norm_poseTheOther_clips[clip_ind,:,1,:]+norm_poseTheOther_clips[clip_ind,:,2,:])/2
        if np.any(np.linalg.norm(head)==0) or np.any(np.linalg.norm(head_TO)==0):
            print('warning: head Length == 0')
            continue


        ### Head Body angle
        cos_angles_hb = []
        sin_angles_hb = []
        angles_hb = []
        for i in range(arg.clip_length):
            angle, cos_angle,sin_angle = angle_between(body[i,:],head[i,:])
            if cos_angle==np.nan or sin_angle==np.nan:
                print('cos_angle==np.nan or sin_angle==np.nan',(body[i,:],head[i,:]))
            cos_angles_hb.append(cos_angle)
            sin_angles_hb.append(sin_angle)
            angles_hb.append(angle)

        cos_angles_hb_TO = []
        sin_angles_hb_TO = []
        angles_hb_TO = []
        for i in range(arg.clip_length):
            angle, cos_angle,sin_angle = angle_between(body_TO[i,:],head_TO[i,:])
            if cos_angle==np.nan or sin_angle==np.nan:
                print('cos_angle==np.nan or sin_angle==np.nan',(body_TO[i,:],head_TO[i,:]))
            cos_angles_hb_TO.append(cos_angle)
            sin_angles_hb_TO.append(sin_angle)
            angles_hb_TO.append(angle)

        ### FFT of nose
        point = 0
        vec = norm_pose_clips[clip_ind,:,point,0] + 1j * norm_pose_clips[clip_ind,:,point,1]
        np_fft = np.fft.fft(vec)
        amplitudes = 2/vec.size * np.abs(np_fft)
        angles = np.angle(np_fft)

        point = 0
        vec_TO = norm_poseTheOther_clips[clip_ind,:,point,0] + 1j * norm_poseTheOther_clips[clip_ind,:,point,1]
        np_fft_TO = np.fft.fft(vec_TO)
        amplitudes_TO = 2/vec_TO.size * np.abs(np_fft_TO)
        angles_TO = np.angle(np_fft)
        
        ### Displacement
        displace_x = norm_pose_clips[clip_ind,:,0,0]
        displace_y = norm_pose_clips[clip_ind,:,0,1]
        displace_x_TO = norm_poseTheOther_clips[clip_ind,:,0,0]
        displace_y_TO = norm_poseTheOther_clips[clip_ind,:,0,1]

        ### Displacement on polar coordinates
        rho_vector = []
        phi_sin_vector = []
        phi_cos_vector = []
        for d_i in range(len(displace_x)):
            d_x = displace_x[d_i]
            d_y = displace_y[d_i]
            rho, phi = cart2pol(d_x, d_y)
            c, s = np.cos(phi), np.sin(phi)
            rho_vector.append(rho)
            phi_sin_vector.append(s)
            phi_cos_vector.append(c)

        rho_vector_TO = []
        phi_sin_vector_TO = []
        phi_cos_vector_TO = []
        for d_i in range(len(displace_x_TO)):
            d_x = displace_x_TO[d_i]
            d_y = displace_y_TO[d_i]
            rho, phi = cart2pol(d_x, d_y)
            c, s = np.cos(phi), np.sin(phi)
            rho_vector_TO.append(rho)
            phi_sin_vector_TO.append(s)
            phi_cos_vector_TO.append(c)


        ### Contour PCA fft
        np_fft_contour = np.fft.fft(cont_clips[clip_ind,:,:],axis = 0)
        amplitudes_contour = np.abs(np_fft_contour)
        amplitudes_contour = amplitudes_contour[0:8,:]
        amplitudes_contour = amplitudes_contour.flatten()
        angles_contour = np.angle(np_fft_contour)
        angles_contour = angles_contour[0:8,:]
        angles_contour = angles_contour.flatten()


        ### Angle of two mouse body
        cos_angles_twoBody = []
        sin_angles_twoBody = []
        angles_hb_twoBody = []
        for i in range(arg.clip_length):
            angle, cos_angle,sin_angle = angle_between(body[i,:],body_TO[i,:])
            if cos_angle==np.nan or sin_angle==np.nan:
                print((body[i,:],head[i,:]))
            cos_angles_twoBody.append(cos_angle)
            sin_angles_twoBody.append(sin_angle)
            angles_hb_twoBody.append(angle)
        
        ### Distance between one mouse’s tail and the other mouse’s nose.
        distance_Tmouse_nose_to_Rmouse_tail_x = norm_poseTheOther_clips[clip_ind,:,3,0]
        distance_Tmouse_nose_to_Rmouse_tail_y = norm_poseTheOther_clips[clip_ind,:,3,1]
        rho_vector_nose_tail = []
        phi_sin_vector_nose_tail = []
        phi_cos_vector_nose_tail = []
        for d_i in range(len(distance_Tmouse_nose_to_Rmouse_tail_x)):
            d_x = distance_Tmouse_nose_to_Rmouse_tail_x[d_i]
            d_y = distance_Tmouse_nose_to_Rmouse_tail_y[d_i]
            rho, phi = cart2pol(d_x, d_y)
            c, s = np.cos(phi), np.sin(phi)
            rho_vector_nose_tail.append(rho)
            phi_sin_vector_nose_tail.append(s)
            phi_cos_vector_nose_tail.append(c)

        body_length_clips.append(np.linalg.norm(body, axis = 1))
        head_earMid_length_clips.append(np.linalg.norm(head, axis = 1))
        head_body_ang_clips.append(np.concatenate((np.asarray(sin_angles_hb),np.asarray(cos_angles_hb))))
        displacementx_clips.append(np.asarray(displace_x))
        displacementy_clips.append(np.asarray(displace_y))
        displace_rho_clips.append(np.asarray(rho_vector))
        displace_phi_s_clips.append(np.asarray(phi_sin_vector))
        displace_phi_c_clips.append(np.asarray(phi_cos_vector))
        
        nose_fft_amp_clips.append(np.asarray(amplitudes[0:8]))
        nose_fft_ang_clips.append(np.concatenate((np.sin(angles)[0:8], np.cos(angles)[0:8])))
        contourPCA_fft_amp_clips.append(np.asarray(amplitudes_contour[0:8]))
        contourPCA_fft_ang_clips.append(np.concatenate((np.sin(angles_contour)[0:8], np.cos(angles_contour)[0:8])))
        
        body_length_clips_TO.append(np.linalg.norm(body_TO, axis = 1))
        head_earMid_length_clips_TO.append(np.linalg.norm(head_TO, axis = 1))
        head_body_ang_clips_TO.append(np.concatenate((np.asarray(sin_angles_hb_TO),np.asarray(cos_angles_hb_TO))))
        displacementx_clips_TO.append(np.asarray(displace_x_TO))
        displacementy_clips_TO.append(np.asarray(displace_y_TO))
        displace_rho_clips_TO.append(np.asarray(rho_vector_TO))
        displace_phi_s_clips_TO.append(np.asarray(phi_sin_vector_TO))
        displace_phi_c_clips_TO.append(np.asarray(phi_cos_vector_TO))
        
        nose_fft_amp_clips_TO.append(np.asarray(amplitudes_TO[0:8]))
        nose_fft_ang_clips_TO.append(np.concatenate((np.sin(angles_TO)[0:8], np.cos(angles_TO)[0:8])))

        two_body_ang_clips.append(\
            np.concatenate((np.asarray(sin_angles_twoBody),\
                            np.asarray(cos_angles_twoBody))))
        nose_tail_displace_rho_clips.append(np.asarray(rho_vector_nose_tail))
        nose_tail_displace_phi_clips.append(
            np.concatenate((np.asarray(phi_sin_vector_nose_tail),\
                            np.asarray(phi_cos_vector_nose_tail))))

        raw_feature_clip_dict={
            'displace_x':displace_x,
            'displace_y':displace_y,
            'displace_rho':rho_vector,
            'displace_phi_s':phi_sin_vector,
            'displace_phi_c':phi_cos_vector,
            'body_length':np.linalg.norm(body, axis = 1),
            'head_length':np.linalg.norm(head, axis = 1),
            'head_body_angles':np.asarray(angles_hb),
            'displace_x_TO':displace_x_TO,
            'displace_y_TO':displace_y_TO,
            'displace_rho_TO':rho_vector_TO,
            'displace_phi_s_TO':phi_sin_vector_TO,
            'displace_phi_c_TO':phi_cos_vector_TO,
            'body_length_TO':np.linalg.norm(body_TO, axis = 1),
            'head_length_TO':np.linalg.norm(head_TO, axis = 1),
            'head_body_angles_TO':np.asarray(angles_hb_TO),
            'cos_angles_twoBody':np.asarray(cos_angles_twoBody),
            'sin_angles_twoBody':np.asarray(sin_angles_twoBody),
        }


        raw_feature_clips_dict.append(raw_feature_clip_dict)
        info_clips_selected.append(info_clips[clip_ind])
        cont_clips_selected.append(cont_clips[clip_ind])
        pose_clips_selected.append(pose_clips[clip_ind])
        poseTheOther_clips_selected.append(poseTheOther_clips[clip_ind])
        # good_index.append(clip_ind)

    # print(displacementx_clips)
    # good_index = np.nonzero(1-bad_clip)
    # good_index = list(good_index[1])
    # good_index = list(good_index[0])
    # print(good_index,good_index[0],len(body_length_clips))
 
    # info_clips = [info_clips[i] for i in good_index]
    # cont_clips = [cont_clips[i] for i in good_index]
    all_info_selected = {
        'info_clips':info_clips_selected,
        'cont_clips':np.asarray(cont_clips_selected),
        'pose_clips':pose_clips_selected,
        'poseTheOther_clips':poseTheOther_clips_selected
    }

    feature_clips_dict={
            'displace_x':displacementx_clips,
            'displace_y':displacementy_clips,
            'displace_rho':displacementx_clips,
            'displace_phi_c':displace_phi_c_clips,
            'displace_phi_s':displace_phi_s_clips,
            'body_length':body_length_clips,
            'head_length':head_earMid_length_clips,
            'head_body_angles':head_body_ang_clips,
            'nose_fft_amp':nose_fft_amp_clips,
            'nose_fft_ang':nose_fft_ang_clips,
            'contourPCA_fft_amp':contourPCA_fft_amp_clips,
            'contourPCA_fft_ang':contourPCA_fft_ang_clips,

            'displace_x_TO':displacementx_clips_TO,
            'displace_y_TO':displacementy_clips_TO,
            'displace_rho_TO':displacementx_clips_TO,
            'displace_phi_c_TO':displace_phi_c_clips_TO,
            'displace_phi_s_TO':displace_phi_s_clips_TO,
            'body_length_TO':body_length_clips_TO,
            'head_length_TO':head_earMid_length_clips_TO,
            'head_body_angles_TO':head_body_ang_clips_TO,
            'nose_fft_amp_TO':nose_fft_amp_clips_TO,
            'nose_fft_ang_TO':nose_fft_ang_clips_TO,

            'two_body_ang':two_body_ang_clips,
            'nose_tail_displace_rho_clips':nose_tail_displace_rho_clips,
            'nose_tail_displace_phi_clips':nose_tail_displace_phi_clips,     
            }

    return feature_clips_dict,all_info_selected,raw_feature_clips_dict

def compute_allAngle(vectors1,vectors2):
    cos_angles = []
    sin_angles = []
    angles = []
    for i in range(vectors1.shape[0]):
        angle, cos_angle,sin_angle = angle_between(vectors1[i,:],vectors2[i,:])
        if cos_angle==np.nan or sin_angle==np.nan:
            print('cos_angle==np.nan or sin_angle==np.nan with:',(vectors1[i,:],vectors2[i,:]))
        cos_angles.append(cos_angle)
        sin_angles.append(sin_angle)
        angles.append(angle)
    return cos_angles, sin_angles, angles

def compute_allDistance(vectorx,vectory):
    distances = []
    for i in range(len(vectorx)):
        dis = (vectorx[i]**2+vectory[i]**2)**0.5
        distances.append(dis)
    return distances

def compute_fft_Nx2(vector_Nx2,first_k=8):
    vec = vector_Nx2[:,0] + 1j * vector_Nx2[:,1]
    np_fft = np.fft.fft(vec)
    amplitudes = 2/vec.size * np.abs(np_fft)
    angles = np.angle(np_fft)
    return amplitudes[0:first_k], angles[0:first_k]

def compute_fft_NxM(mtr,first_k=-1):
    np_fft_contour = np.fft.fft(mtr,axis = 0)
    amplitudes_contour = np.abs(np_fft_contour)
    amplitudes_contour = amplitudes_contour[0:first_k,:]
    amplitudes_contour = amplitudes_contour.flatten()
    angles_contour = np.angle(np_fft_contour)
    angles_contour = angles_contour[0:first_k,:]
    angles_contour = angles_contour.flatten()

    return amplitudes_contour,angles_contour

def toPolarCoordinates(displace_x,displace_y):
    rho_vector = []
    phi_sin_vector = []
    phi_cos_vector = []
    for d_i in range(len(displace_x)):
        d_x = displace_x[d_i]
        d_y = displace_y[d_i]
        rho, phi = cart2pol(d_x, d_y)
        c, s = np.cos(phi), np.sin(phi)
        rho_vector.append(rho)
        phi_sin_vector.append(s)
        phi_cos_vector.append(c)
    return rho_vector, phi_sin_vector, phi_cos_vector

def compute_features_sep_twoMice_Independent(arg,clips_dict):
    pose_clips_align = clips_dict['pose_clips_align']
    poseTheOther_clips_alignSelf = clips_dict['poseTheOther_clips_alignSelf']
    poseTheOther_clips_alignToOther = clips_dict['poseTheOther_clips_alignToOther']
    cont_clips = clips_dict['cont_clips']
    contTheOther_clips = clips_dict['contTheOther_clips']
    pose_clips = clips_dict['pose_clips']
    poseTheOther_clips = clips_dict['poseTheOther_clips']
    info_clips = clips_dict['info_clips']
    frames_path_clips = clips_dict['frames_path_clips']
    clip_num,frame_num,point_num,_ = pose_clips_align.shape
    raw_feature_clips = []
    raw_feature_clips_dict = []
    feature_clips_dict = {}

    body_length_clips = []
    head_earMid_length_clips = []
    head_body_ang_clips = []
    displacementx_clips = []
    displacementy_clips = []
    displace_rho_clips = []
    displace_phi_s_clips = []
    displace_phi_c_clips = []
    nose_fft_amp_clips = []
    nose_fft_ang_clips = []
    contourPCA_fft_amp_clips = []
    contourPCA_fft_ang_clips = []

    body_length_clips_TO = []
    head_earMid_length_clips_TO = []
    head_body_ang_clips_TO = []
    displacementx_clips_TO = []
    displacementy_clips_TO = []
    displace_rho_clips_TO = []
    displace_phi_s_clips_TO = []
    displace_phi_c_clips_TO = []
    nose_fft_amp_clips_TO = []
    nose_fft_ang_clips_TO = []
    contourPCA_fft_amp_clips_TO = []
    contourPCA_fft_ang_clips_TO = []

    two_body_ang_clips = []
    two_head_ang_clips = []
    body_change_ang_clips = []
    TM_nose_RM_tail_displace_rho_clips = []
    TM_nose_RM_tail_displace_phi_clips = []
    RM_nose_TM_tail_displace_rho_clips = []
    RM_nose_TM_tail_displace_phi_clips = []
    nose_nose_displace_rho_clips = []
    nose_nose_displace_phi_clips = []
    TM_nose_RM_tail_distance_clips = []
    RM_nose_TM_tail_distance_clips = []
    nose_nose_distance_clips = []

    info_clips_selected = []
    frames_path_clips_selected = []
    cont_clips_selected = []
    contTheOther_clips_selected = []
    pose_clips_selected = []
    poseTheOther_clips_selected = []

    newFeatureName_clip = []

    good_index = []
    number_of_clip_used = 0
    for clip_ind in range(clip_num):
        ### Body vector,  eventually Body Length is used as feature
        body = pose_clips_align[clip_ind,:,0,:] - pose_clips_align[clip_ind,:,3,:]
        body_TO = poseTheOther_clips_alignSelf[clip_ind,:,0,:] - poseTheOther_clips_alignSelf[clip_ind,:,3,:]
        if  np.any(np.linalg.norm(body)==0) or np.any(np.linalg.norm(body_TO)==0):
            print('warning: Body Length == 0')
            continue
       
        ### Head ear-middle-point vector, eventually head Length is used as feature
        head = pose_clips_align[clip_ind,:,0,:] - (pose_clips_align[clip_ind,:,1,:]+pose_clips_align[clip_ind,:,2,:])/2
        head_TO = poseTheOther_clips_alignSelf[clip_ind,:,0,:] - (poseTheOther_clips_alignSelf[clip_ind,:,1,:]+poseTheOther_clips_alignSelf[clip_ind,:,2,:])/2
        if np.any(np.linalg.norm(head)==0) or np.any(np.linalg.norm(head_TO)==0):
            print('warning: head Length == 0')
            continue

        ### Head Body angle
        cos_angles_hb, sin_angles_hb, angles_hb = compute_allAngle(body,head)
        cos_angles_hb_TO, sin_angles_hb_TO, angles_hb_TO = compute_allAngle(body_TO,head_TO)
        
        ### FFT of nose
        amplitudes, angles = compute_fft_Nx2(pose_clips_align[clip_ind,:,0,:],first_k=arg.fft_firstK)
        amplitudes_TO, angles_TO = compute_fft_Nx2(poseTheOther_clips_alignSelf[clip_ind,:,0,:],first_k=arg.fft_firstK)
       
        ### Displacement
        displace_x, displace_y = pose_clips_align[clip_ind,:,0,0], pose_clips_align[clip_ind,:,0,1]
        displace_x_TO, displace_y_TO = poseTheOther_clips_alignSelf[clip_ind,:,0,0], poseTheOther_clips_alignSelf[clip_ind,:,0,1]

        ### Displacement on polar coordinates
        rho_vector, phi_sin_vector, phi_cos_vector = toPolarCoordinates(displace_x,displace_y)
        rho_vector_TO, phi_sin_vector_TO, phi_cos_vector_TO = toPolarCoordinates(displace_x_TO,displace_y_TO)

        ### Contour PCA fft
        amplitudes_contour,angles_contour = compute_fft_NxM(cont_clips[clip_ind,:,:],first_k=arg.fft_firstK)
        amplitudes_contour_TO,angles_contour_TO = compute_fft_NxM(contTheOther_clips[clip_ind,:,:],first_k=arg.fft_firstK)

        ### Angle of two mouse body
        body_noAlign = pose_clips[clip_ind,:,0,:] - pose_clips[clip_ind,:,3,:]
        body_noAlign_TO = poseTheOther_clips[clip_ind,:,0,:] - poseTheOther_clips[clip_ind,:,3,:]
        cos_angles_twoBody, sin_angles_twoBody, angles_twoBody = \
            compute_allAngle(body_noAlign,body_noAlign_TO)

        ### Angle of two mouse head
        head_noAlign = pose_clips[clip_ind,:,0,:] - (pose_clips[clip_ind,:,1,:]+pose_clips[clip_ind,:,2,:])/2
        head_noAlign_TO = poseTheOther_clips[clip_ind,:,0,:] - (poseTheOther_clips[clip_ind,:,1,:]+poseTheOther_clips[clip_ind,:,2,:])/2
        cos_angles_twoHead, sin_angles_twoHead, angles_twoHead = \
            compute_allAngle(head_noAlign,head_noAlign_TO)
        
        ### Angle of current body and the previous body
        body = pose_clips[clip_ind,:,0,:] - pose_clips[clip_ind,:,3,:]
        cos_angles_bodyChange = []
        sin_angles_bodyChange = []
        for i in range(1, arg.clip_length):
            angle, cos_angle,sin_angle = angle_between(body[i-1,:],body[i,:])
            if cos_angle==np.nan or sin_angle==np.nan:
                print((body[i,:],head[i,:]))
            cos_angles_bodyChange.append(cos_angle)
            sin_angles_bodyChange.append(sin_angle)
        if 'body_change_ang' in feature_clips_dict:
            feature_clips_dict['body_change_ang'].append(np.concatenate((np.asarray(sin_angles_bodyChange),np.asarray(cos_angles_bodyChange))))
        else:
            feature_clips_dict['body_change_ang'] = [np.concatenate((np.asarray(sin_angles_bodyChange),np.asarray(cos_angles_bodyChange)))]

       
        ### Displacement from  target mouse’s nose to refer mouse’s tail .
        displace_Tmouse_nose_to_Rmouse_tail_x = poseTheOther_clips_alignToOther[clip_ind,:,3,0]
        displace_Tmouse_nose_to_Rmouse_tail_y = poseTheOther_clips_alignToOther[clip_ind,:,3,1]
        rho_vector_nose_tail_0, phi_sin_vector_nose_tail_0, phi_cos_vector_nose_tail_0 = \
            toPolarCoordinates(displace_Tmouse_nose_to_Rmouse_tail_x,displace_Tmouse_nose_to_Rmouse_tail_y)
        
        ### Displacement from  refer mouse’s nose to target mouse’s tail .
        displace_Rmouse_nose_to_Tmouse_tail_x = poseTheOther_clips_alignToOther[clip_ind,:,0,0] + pose_clips_align[clip_ind,:,3,0]
        displace_Rmouse_nose_to_Tmouse_tail_y = poseTheOther_clips_alignToOther[clip_ind,:,0,1] + pose_clips_align[clip_ind,:,3,1]
        rho_vector_nose_tail_1, phi_sin_vector_nose_tail_1, phi_cos_vector_nose_tail_1 = \
            toPolarCoordinates(displace_Rmouse_nose_to_Tmouse_tail_x,displace_Rmouse_nose_to_Tmouse_tail_y)
        
        ### Displacement between one mouse’s nose and the other mouse’s nose.
        displace_Tmouse_nose_to_Rmouse_nose_x = poseTheOther_clips_alignToOther[clip_ind,:,0,0]
        displace_Tmouse_nose_to_Rmouse_nose_y = poseTheOther_clips_alignToOther[clip_ind,:,0,1]
        rho_vector_nose_nose, phi_sin_vector_nose_nose, phi_cos_vector_nose_nose = \
            toPolarCoordinates(displace_Tmouse_nose_to_Rmouse_nose_x,displace_Tmouse_nose_to_Rmouse_nose_y)

        ## you can process data to get the feature
        ## the pose data are in five variables: pose_clips, pose_clips_align, poseTheOther_clips, poseTheOther_clips_alignSelf, poseTheOther_clips_alignToOther 
        ## each of the variables is a numpy array whose shape is (number_of_clip, number_of_frames_in_one_clip, number_of_key_point, 3)
        ## pose_clips contains the raw key points data of the target mouse
        ## pose_clips_align contains key point data that is aligned to the target mouse in the middle frame 
        ## poseTheOther_clips contains the raw key points data of the mouse that is closest to the target mouse
        ## poseTheOther_clips_alignSelf contains the key points data of the closest mousethat that is aligned to itself in the middle frame 
        ## poseTheOther_clips_alignToOther contains the key points data of the closest mousethat that is aligned to the target mouse in the middle frame 
        newFeature = np.ones(pose_clips.shape[1])  # the feature of one clip should be a numpy array whose shape is the frame number of the clip 
        if 'newFeatureName' in feature_clips_dict:
            feature_clips_dict['newFeatureName'].append(newFeature)
        else:
            feature_clips_dict['newFeatureName'] = [newFeature]

            
        
        body_length_clips.append(np.linalg.norm(body, axis = 1))
        head_earMid_length_clips.append(np.linalg.norm(head, axis = 1))
        head_body_ang_clips.append(np.concatenate((np.asarray(sin_angles_hb),np.asarray(cos_angles_hb))))
        displacementx_clips.append(np.asarray(displace_x))
        displacementy_clips.append(np.asarray(displace_y))
        displace_rho_clips.append(np.asarray(rho_vector))
        displace_phi_s_clips.append(np.asarray(phi_sin_vector))
        displace_phi_c_clips.append(np.asarray(phi_cos_vector)) 
        nose_fft_amp_clips.append(np.asarray(amplitudes))
        nose_fft_ang_clips.append(np.concatenate((np.sin(angles), np.cos(angles))))
        contourPCA_fft_amp_clips.append(np.asarray(amplitudes_contour))
        contourPCA_fft_ang_clips.append(np.concatenate((np.sin(angles_contour), np.cos(angles_contour))))
        
        body_length_clips_TO.append(np.linalg.norm(body_TO, axis = 1))
        head_earMid_length_clips_TO.append(np.linalg.norm(head_TO, axis = 1))
        head_body_ang_clips_TO.append(np.concatenate((np.asarray(sin_angles_hb_TO),np.asarray(cos_angles_hb_TO))))
        displacementx_clips_TO.append(np.asarray(displace_x_TO))
        displacementy_clips_TO.append(np.asarray(displace_y_TO))
        displace_rho_clips_TO.append(np.asarray(rho_vector_TO))
        displace_phi_s_clips_TO.append(np.asarray(phi_sin_vector_TO))
        displace_phi_c_clips_TO.append(np.asarray(phi_cos_vector_TO))
        nose_fft_amp_clips_TO.append(np.asarray(amplitudes_TO))
        nose_fft_ang_clips_TO.append(np.concatenate((np.sin(angles_TO), np.cos(angles_TO))))
        contourPCA_fft_amp_clips_TO.append(np.asarray(amplitudes_contour_TO))
        contourPCA_fft_ang_clips_TO.append(np.concatenate((np.sin(angles_contour_TO), np.cos(angles_contour_TO))))
        

        two_body_ang_clips.append(\
            np.concatenate((np.asarray(sin_angles_twoBody),\
                            np.asarray(cos_angles_twoBody))))
        two_head_ang_clips.append(\
            np.concatenate((np.asarray(sin_angles_twoHead),\
                            np.asarray(cos_angles_twoHead))))

        TM_nose_RM_tail_displace_rho_clips.append(np.asarray(rho_vector_nose_tail_0))
        TM_nose_RM_tail_displace_phi_clips.append(
            np.concatenate((np.asarray(phi_sin_vector_nose_tail_0),\
                            np.asarray(phi_cos_vector_nose_tail_0))))
        RM_nose_TM_tail_displace_rho_clips.append(np.asarray(rho_vector_nose_tail_1))
        RM_nose_TM_tail_displace_phi_clips.append(
            np.concatenate((np.asarray(phi_sin_vector_nose_tail_1),\
                            np.asarray(phi_cos_vector_nose_tail_1))))
        nose_nose_displace_rho_clips.append(np.asarray(rho_vector_nose_nose))
        nose_nose_displace_phi_clips.append(
            np.concatenate((np.asarray(phi_sin_vector_nose_nose),\
                            np.asarray(phi_cos_vector_nose_nose))))
        TM_nose_RM_tail_distance_clips.append(
            compute_allDistance(displace_Tmouse_nose_to_Rmouse_tail_x, displace_Tmouse_nose_to_Rmouse_tail_y))
        RM_nose_TM_tail_distance_clips.append(
            compute_allDistance(displace_Rmouse_nose_to_Tmouse_tail_x, displace_Rmouse_nose_to_Tmouse_tail_y))
        nose_nose_distance_clips.append(
            compute_allDistance(displace_Tmouse_nose_to_Rmouse_nose_x, displace_Tmouse_nose_to_Rmouse_nose_y))

        raw_feature_clip_dict={
            'displace_x':displace_x,
            'displace_y':displace_y,
            'displace_rho':rho_vector,
            'displace_phi_s':phi_sin_vector,
            'displace_phi_c':phi_cos_vector,
            'body_length':np.linalg.norm(body, axis = 1),
            'head_length':np.linalg.norm(head, axis = 1),
            'head_body_angles':np.asarray(angles_hb),
            'displace_x_TO':displace_x_TO,
            'displace_y_TO':displace_y_TO,
            'displace_rho_TO':rho_vector_TO,
            'displace_phi_s_TO':phi_sin_vector_TO,
            'displace_phi_c_TO':phi_cos_vector_TO,
            'body_length_TO':np.linalg.norm(body_TO, axis = 1),
            'head_length_TO':np.linalg.norm(head_TO, axis = 1),
            'head_body_angles_TO':np.asarray(angles_hb_TO),
            'cos_angles_twoBody':np.asarray(cos_angles_twoBody),
            'sin_angles_twoBody':np.asarray(sin_angles_twoBody),
            'cos_angles_twoHead':np.asarray(cos_angles_twoHead),
            'sin_angles_twoHead':np.asarray(sin_angles_twoHead),
            'cos_angles_bodyChange':np.asarray(cos_angles_bodyChange),
            'sin_angles_bodyChange':np.asarray(sin_angles_bodyChange),

        }

        raw_feature_clips_dict.append(raw_feature_clip_dict)
        info_clips_selected.append(info_clips[clip_ind])
        frames_path_clips_selected.append(frames_path_clips[clip_ind])
        cont_clips_selected.append(cont_clips[clip_ind])
        contTheOther_clips_selected.append(contTheOther_clips[clip_ind])
        pose_clips_selected.append(pose_clips[clip_ind])
        poseTheOther_clips_selected.append(poseTheOther_clips[clip_ind])

        number_of_clip_used+=1
        if number_of_clip_used>arg.max_clips_num:
            print('max_clips_num is set to {} will stop processing more clips'.format(arg.max_clips_num))
            break
        
    all_info_selected = {
        'info_clips':info_clips_selected,
        'frames_path_clips':frames_path_clips_selected,
        'cont_clips':np.asarray(cont_clips_selected),
        'contTheOther_clips':np.asarray(contTheOther_clips_selected),
        'pose_clips':pose_clips_selected,
        'poseTheOther_clips':poseTheOther_clips_selected
    }

    feature_clips_dict.update({
            'displace_x':displacementx_clips,
            'displace_y':displacementy_clips,
            'displace_rho':displacementx_clips,
            'displace_phi_c':displace_phi_c_clips,
            'displace_phi_s':displace_phi_s_clips,
            'body_length':body_length_clips,
            'head_length':head_earMid_length_clips,
            'head_body_angles':head_body_ang_clips,
            'nose_fft_amp':nose_fft_amp_clips,
            'nose_fft_ang':nose_fft_ang_clips,
            'contourPCA_fft_amp':contourPCA_fft_amp_clips,
            'contourPCA_fft_ang':contourPCA_fft_ang_clips,

            'displace_x_TO':displacementx_clips_TO,
            'displace_y_TO':displacementy_clips_TO,
            'displace_rho_TO':displacementx_clips_TO,
            'displace_phi_c_TO':displace_phi_c_clips_TO,
            'displace_phi_s_TO':displace_phi_s_clips_TO,
            'body_length_TO':body_length_clips_TO,
            'head_length_TO':head_earMid_length_clips_TO,
            'head_body_angles_TO':head_body_ang_clips_TO,
            'nose_fft_amp_TO':nose_fft_amp_clips_TO,
            'nose_fft_ang_TO':nose_fft_ang_clips_TO,
            'contourPCA_fft_amp_TO':contourPCA_fft_amp_clips_TO,
            'contourPCA_fft_ang_TO':contourPCA_fft_ang_clips_TO,

            'two_body_ang':two_body_ang_clips,
            'two_head_ang':two_head_ang_clips,
            # 'body_change_ang':body_change_ang_clips,
            'TM_nose_RM_tail_displace_phi':TM_nose_RM_tail_displace_phi_clips,
            'TM_nose_RM_tail_displace_rho':TM_nose_RM_tail_displace_rho_clips,
            'RM_nose_TM_tail_displace_phi':RM_nose_TM_tail_displace_phi_clips,
            'RM_nose_TM_tail_displace_rho':RM_nose_TM_tail_displace_rho_clips,
            'nose_nose_displace_rho':nose_nose_displace_rho_clips,     
            'nose_nose_displace_phi':nose_nose_displace_phi_clips,
            'TM_nose_RM_tail_distance':TM_nose_RM_tail_distance_clips,
            'RM_nose_TM_tail_distance':RM_nose_TM_tail_distance_clips, 
            'nose_nose_distance':nose_nose_distance_clips,   
        }
    )

    return feature_clips_dict,all_info_selected,raw_feature_clips_dict

def compute_features(arg,norm_pose_clips,info_clips,cont_clips):

    clip_num,frame_num,point_num,_ = norm_pose_clips.shape
    feature_clips = []
    raw_feature_clips = []

    bad_clip = np.zeros((1,clip_num))
    for clip_ind in range(clip_num):

        ## FFT of nose
        point = 0
        vec = norm_pose_clips[clip_ind,:,point,0] + 1j * norm_pose_clips[clip_ind,:,point,1]
        np_fft = np.fft.fft(vec)
        amplitudes = 2/vec.size * np.abs(np_fft)
        angles = np.angle(np_fft)

        feature_clip = np.concatenate((amplitudes[0:8], np.cos(angles)[0:8]))
        feature_clip = np.concatenate((feature_clip,np.sin(angles)[0:8]))
        raw_feature_clip = np.concatenate((norm_pose_clips[clip_ind,:,point,0],norm_pose_clips[clip_ind,:,point,1]))


        ## Body Length
        mid_id = arg.clip_length // 2 +1
        mid_pose = norm_pose_clips[clip_ind,mid_id,:,:]

        mid_nose,mid_ear1,mid_ear2,mid_tail = mid_pose[0,0:2], mid_pose[1,0:2], mid_pose[2,0:2], mid_pose[3,0:2]

        mid_body = mid_nose-mid_tail
        if np.linalg.norm(mid_body) == 0:
            bad_clip[clip_ind] = 1
            continue

        mid_head = mid_nose - (mid_ear1+mid_ear2)/2
        if np.linalg.norm(mid_head) == 0:
            bad_clip[clip_ind] = 1
            continue

        body = norm_pose_clips[clip_ind,:,0,:] - norm_pose_clips[clip_ind,:,3,:]
        if np.any(np.linalg.norm(body)==0):
            bad_clip[clip_ind] = 1
            continue

        feature_clip = np.concatenate((feature_clip,np.linalg.norm(body, axis = 1)))
        raw_feature_clip = np.concatenate((raw_feature_clip,np.linalg.norm(body, axis = 1)))

        ### Head Body angle
        head = norm_pose_clips[clip_ind,:,0,:] - (norm_pose_clips[clip_ind,:,1,:]+norm_pose_clips[clip_ind,:,2,:])/2
        if np.any(np.linalg.norm(head)==0):
            bad_clip[clip_ind] = 1
            continue

        feature_clip = np.concatenate((feature_clip,np.linalg.norm(head, axis = 1)))
        raw_feature_clip = np.concatenate((raw_feature_clip,np.linalg.norm(head, axis = 1)))

        cos_angles = []
        sin_angles = []
        angles = []
        for i in range(arg.clip_length):
            angle, cos_angle,sin_angle = angle_between(body[i,:],head[i,:])
            cos_angles.append(cos_angle)
            sin_angles.append(sin_angle)
            angles.append(angle)

        feature_clip = np.concatenate((feature_clip,np.asarray(cos_angles)))
        feature_clip = np.concatenate((feature_clip,np.asarray(sin_angles)))
        raw_feature_clip = np.concatenate((raw_feature_clip,np.asarray(angles)))

        ### Displacement angle
        displace_x = norm_pose_clips[clip_ind,:,0,0]
        displace_y = norm_pose_clips[clip_ind,:,0,1]
        feature_clip = np.concatenate((feature_clip,displace_x))
        feature_clip = np.concatenate((feature_clip,displace_y))
        raw_feature_clip = np.concatenate((raw_feature_clip,displace_x))
        raw_feature_clip = np.concatenate((raw_feature_clip,displace_y))

        ## Contour PCA fft
        np_fft = np.fft.fft(cont_clips[clip_ind,:,:],axis = 0)
        amplitudes = np.abs(np_fft)
        amplitudes = amplitudes[0:8,:]
        amplitudes = amplitudes.flatten()

        angles = np.angle(np_fft)
        angles = angles[0:8,:]
        angles = angles.flatten()
        feature_clip = np.concatenate((feature_clip,amplitudes))


        feature_clip = np.concatenate((feature_clip,np.cos(angles)))
        feature_clip = np.concatenate((feature_clip,np.sin(angles)))

        feature_clips.append(feature_clip)
        raw_feature_clips.append(raw_feature_clip)

    good_index = np.nonzero(1-bad_clip)
    good_index = list(good_index[1])

    feature_clips = [feature_clips[i] for i in good_index]
    info_clips = [info_clips[i] for i in good_index]
    cont_clips = [cont_clips[i] for i in good_index]

    return feature_clips,raw_feature_clips,info_clips,cont_clips

def normalize_features(feature_clips):

    temp1 = feature_clips[:, 0:8]
    feature_clips[:, 0:8] =   (temp1-np.mean(temp1))/np.std(temp1)

    temp2 = feature_clips[:, 8:16]
    feature_clips[:, 8:16] =  (temp2-np.mean(temp2))/np.std(temp2)

    temp3 = feature_clips[:, 16:24]
    feature_clips[:, 16:24] = (temp3-np.mean(temp3))/np.std(temp3)

    temp4 = feature_clips[:, 24:39]
    feature_clips[:, 24:39] = (temp4-np.mean(temp4))/np.std(temp4)*3

    temp5 = feature_clips[:, 39:54]
    feature_clips[:, 39:54] = (temp5-np.mean(temp5))/np.std(temp5)*3

    temp6 = feature_clips[:, 54:69]
    feature_clips[:, 54:69] = (temp6-np.mean(temp6))/np.std(temp6)*2

    temp7 = feature_clips[:, 69:84]
    feature_clips[:, 69:84] = (temp7-np.mean(temp7))/np.std(temp7)*2

    temp8 = feature_clips[:, 84:99]
    feature_clips[:, 84:99] = (temp8-np.mean(temp8))/np.std(temp8)*4

    temp9 = feature_clips[:, 99:114]
    feature_clips[:, 99:114] = (temp9-np.mean(temp9))/np.std(temp9)*4

    temp10 = feature_clips[:, 114:194]
    feature_clips[:, 114:194] = (temp10-np.mean(temp10))/np.std(temp10)*0.8

    temp11 = feature_clips[:, 194:274]
    feature_clips[:, 194:274] = (temp11-np.mean(temp11))/np.std(temp11)*0.3

    temp12 = feature_clips[:, 274:354]
    feature_clips[:, 274:354] = (temp12-np.mean(temp12))/np.std(temp12)*0.3


    print(feature_clips.shape)


    plt.figure()
    sns.heatmap(feature_clips)
    plt.savefig('temporal1.png')
    return feature_clips

def maxmin_norm(feat):
    maxcols=feat.max(axis=0)
    mincols=feat.min(axis=0)
    data_shape = feat.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        if(maxcols[i]==0 and mincols[i]==0):
            continue
        assert maxcols[i]-mincols[i]!=0, "%i,%f,%f"%(i,maxcols[i],mincols[i])
        t[:,i]=(feat[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

def zscore_norm(feat,mean=-1,std=-1):
    if(mean==-1):
        mean=feat.mean(axis=0)
    if(std==-1):
        std=feat.std(axis=0)
    data_shape = feat.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        if(std[i]==0):
            continue
        t[:,i]=(feat[:,i]-mean[i])/std[i]
    return t

def midFrame_div_norm(feat):
    ## this norm is mainly for body length
    data_shape = feat.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    # mid_frame_ind = 7
    mid_frame_ind = int(data_cols/2)
    div_value = feat[:,mid_frame_ind]
    t=np.empty((data_rows,data_cols))
    for i in range(data_rows):
        t[i,:]=feat[i,:]/mid_frame_ind[i]
    return t

def midFrame_minu_norm(feat):
    ## this norm is mainly for body length
    data_shape = feat.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    # mid_frame_ind = 7
    mid_frame_ind = int(data_cols/2)
    div_value = feat[:,mid_frame_ind]
    t=np.empty((data_rows,data_cols))
    for i in range(data_rows):
        t[i,:]=feat[i,:]-mid_frame_ind[i]
    return t

def maxminAll_norm(feat):
    div = (np.max(feat)-np.min(feat))
    div = np.asarray(div,dtype='float')
    div[div==0]=0.0001
    feat =  (feat-np.min(feat))/div
    return feat

def zscoreAll_norm(feat):
    div = np.std(feat)
    div = np.asarray(div,dtype='float')
    div[div==0]=0.0001
    feat =  (feat-np.mean(feat))/div
    return feat

def name_norm(feat,norm_form):
    if norm_form=='no':
        return np.asarray(feat)
    elif norm_form=='zscore_all':
        return zscoreAll_norm(feat)
    elif norm_form=='maxmin_all':
        return maxminAll_norm(feat)
    elif norm_form=='maxmin':
        return maxmin_norm(feat)
    elif norm_form=='zscore':
        return zscore_norm(feat)
    elif norm_form=='zscore_all+maxmin_all':
        return maxminAll_norm(zscoreAll_norm(feat))
    else:
        raise

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def visualize_one_clip(arg,clip_ind,pose_clips,info_clips):

    print('hellp')
    out_video_path = 'results/clip_{}.mp4'.format(clip_ind)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 9.0, (400,400))
    # video_ind,first_frame,last_frame = retrieve_ind(arg,clip_ind)

    pose = pose_clips[clip_ind][7]
    # print(pose)

    nose,ear1,ear2,tail = pose[0,:],pose[1,:],pose[2,:],pose[3,:]
    body = tail-nose
    rho,phi = cart2pol(body[0],body[1])
    angle = math.degrees(phi)

    # print(angle)

    for frame_ind in range(arg.clip_length):

        # print(info_clips[clip_ind][frame_ind])
        img = cv2.imread(info_clips[clip_ind][frame_ind])
        rows,cols,depth = img.shape

        plt.figure()
        plt.plot(nose[0],nose[1],'ro')
        plt.plot(tail[0],tail[1],'bo')
        plt.plot(ear1[0],ear1[1],'go')
        plt.plot(ear2[0],ear2[1],'go')
        plt.savefig('temp.png')
        plt.close()
        M = np.float32([[1,0,640-nose[0]],[0,1,360-nose[1]]])
        tra = cv2.warpAffine(img,M,(cols,rows))


        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        rot = cv2.warpAffine(tra,M,(cols,rows))


        img = rot[160:560,440:840].copy()

        # cv2.putText(img,'cluster_{}_clip_{}'.format(cluster_ind,clip_ind),
        #     arg.bottomLeftCornerOfText,
        #     arg.font,
        #     arg.fontScale,
        #     arg.fontColor,
        #     arg.lineType)

            # dend = cv2.imread('dendrogram.png')
            # dend = cv2.resize(dend,(800,400))
            # img = cv2.hconcat((crop,dend))

        out.write(img)

    out.release()

def visualize_one_cluster(arg,cluster_ind,cluster,g0,f,pose_clips,info_clips,cont_clips,raw_feature_clips,video_name_suffix=''):

    import pickle
    with open('leaves.pckl','rb') as ff:
        leaves = pickle.load(ff)

    out_video_path = 'results/cluster_{}_{}.mp4'.format(video_name_suffix,cluster_ind)
    print(out_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 9.0, (1200,800))

    xlim  = g0.get_xlim()
    xlim = xlim[1]

    # for clip_ind in cluster[0:min(50,len(cluster))]:
    for clip_ind in tqdm(cluster):
        xx = np.squeeze(np.argwhere(leaves == clip_ind))
        xxx = xlim * xx/leaves.size

        ll = g0.plot([xxx,xxx],[0,arg.threshold],linewidth = 4,color = 'red')
        f.savefig('dendrogram.png')

        l = ll[0]
        l.remove()
        del l
        plt.close()


        fig,ax = plt.subplots()
        sns.heatmap(cont_clips[clip_ind,:,:])
        fig.savefig('cont_temp.png')
        plt.close()

        fig,axes = plt.subplots(5,1,figsize= (5,10))
        axes[0].plot(raw_feature_clips[clip_ind,30:45])
        axes[0].set_ylim([80,200])
        axes[0].set_ylabel('Body Length')
        axes[1].plot(raw_feature_clips[clip_ind,45:60])
        axes[1].set_ylim([0,80])
        axes[1].set_ylabel('Head Length')
        axes[2].plot(raw_feature_clips[clip_ind,60:75])
        axes[2].set_ylim([-1.7,1.7])
        axes[2].set_ylabel('Head Body Angle')
        axes[3].plot(raw_feature_clips[clip_ind,75:90])
        axes[3].set_ylim([-60,60])
        axes[3].set_ylabel('Displacement X')
        axes[4].plot(raw_feature_clips[clip_ind,90:105])
        axes[4].set_ylim([-40,40])
        axes[4].set_ylabel('Displacement Y')
        fig.savefig('raw_temp.png')
        plt.close()

        pose = pose_clips[clip_ind][7]
        nose,tail = pose[0,:],pose[3,:]
        body = tail-nose
        rho,phi = cart2pol(body[0],body[1])
        angle = math.degrees(phi)

        for frame_ind in range(arg.clip_length):

            # print(clip_ind,frame_ind)
            img = cv2.imread(info_clips[clip_ind][frame_ind])
            try:
                rows,cols,depth = img.shape
            except Exception as e:
                print(info_clips[clip_ind][frame_ind])
                print(clip_ind,frame_ind)
                raise e
            

            M = np.float32([[1,0,640-nose[0]],[0,1,360-nose[1]]])
            tra = cv2.warpAffine(img,M,(cols,rows))


            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            rot = cv2.warpAffine(tra,M,(cols,rows))


            crop = rot[160:560,440:840].copy()

            cv2.putText(crop,'cluster_{}_clip_{}'.format(cluster_ind,clip_ind),
                arg.bottomLeftCornerOfText,
                arg.font,
                arg.fontScale,
                arg.fontColor,
                arg.lineType)


            cont = cv2.imread('cont_temp.png')
            cont = cv2.resize(cont,(400,400))

            img = cv2.hconcat((crop,cont))

            dend = cv2.imread('dendrogram.png')
            dend = cv2.resize(dend,(800,400))

            img = cv2.vconcat((img,dend))

            raw = cv2.imread('raw_temp.png')
            raw = cv2.resize(raw,(400,800))

            img = cv2.hconcat((img,raw))
            out.write(img)


    out.release()

def sort_cluster(elem):

    import pickle
    with open('leaves.pckl','rb') as f:
        leaves = pickle.load(f)
        # print('help')
    temp = np.nonzero(leaves == elem)
    return temp[0]

def sort_accor_leaves(clusters,leaves):

    # # print(leaves)
    # import pickle
    # with open('leaves.pckl','wb') as f:
    #     # print('lalala')
    #     pickle.dump(leaves,f)

    for cluster_ind in range(len(clusters)):

        cluster = clusters[cluster_ind]
        # print(cluster)
        # print(sort_cluster(2))
        # print(sort_cluster(33))
        # print(sort_cluster(35))
        # cluster = sorted(cluster,key = sort_cluster)
        cluster = sorted(cluster,key = lambda elem: np.nonzero(leaves == elem)[0])
        # print(cluster)
        clusters[cluster_ind] = cluster


    return clusters

def cluster(arg,feature_clips_dict,all_info_selected,raw_feature_clips_dict):
    print('number of clips will used after preprocess: {}'.format(len(feature_clips_dict['displace_x'])))
    all_result = []
    all_result_dict = {}

    for c_arg_dict_idx in range(len(arg.cluster_arg)):
        c_arg_dict = arg.cluster_arg[c_arg_dict_idx]
        print('clustering with {}, with thred:{}'.format(c_arg_dict['name'],c_arg_dict['thred']))
        ### Normalize and concatenate feature  
        feat_norm = c_arg_dict['features_arg'][0]['weight']*get_normFeature_byArg(c_arg_dict['features_arg'][0],feature_clips_dict)
        for f_i in range(1,len(c_arg_dict['features_arg'])):
            feat_norm_tmp = c_arg_dict['features_arg'][f_i]['weight']*get_normFeature_byArg(c_arg_dict['features_arg'][f_i],feature_clips_dict)
            feat_norm = np.concatenate([feat_norm,feat_norm_tmp] ,axis = 1)

        clip_num,fea_num = feat_norm.shape

        ## Perform hierarchical/agglomerative clustering.
        Z = linkage(feat_norm, method ='ward',metric='euclidean')

        # ## Plots the hierarchical clustering as a dendrogram
        dn = hierarchy.dendrogram(Z,orientation='top')
        leaves = np.asarray(dn['leaves'])

        ## Form flat clusters from the hierarchical clustering defined by the given linkage matrix.
        if isinstance(c_arg_dict['thred'],list):
            print('selecting fcluster threshold with %s metric:'%(c_arg_dict['evaluation_metric']))
            score_select = -100000
            thred_select = 0
            for threshold in c_arg_dict['thred']:
                cluster_result_tmp = hierarchy.fcluster(Z,threshold,'distance')
                if c_arg_dict['evaluation_metric'] == 'Davies-Bouldin index':
                    score = metrics.davies_bouldin_score(feat_norm, cluster_result_tmp)
                    if -score > score_select:
                        score_select = -score
                        cluster_result = cluster_result_tmp
                        thred_select = threshold
                elif c_arg_dict['evaluation_metric'] == 'Calinski-Harabasz Index':
                    score = metrics.calinski_harabasz_score(feat_norm, cluster_result_tmp)
                    if score > score_select:
                        score_select = score
                        cluster_result = cluster_result_tmp
                        thred_select = threshold
                elif c_arg_dict['evaluation_metric'] == 'Adjusted Rand index':
                    clusters = [[] for i in range(np.max(cluster_result_tmp)+1)]
                    for ii in range(len(cluster_result_tmp)):
                        clusters[cluster_result_tmp[ii]].append(ii)
                    clusters = sort_accor_leaves(clusters,leaves)
                    print('number of clusters:%d'%(np.max(cluster_result_tmp)))
                    all_result_tmp = [{'clusters':clusters,\
                           'Z':Z,\
                           'cluster_result_fcluster':cluster_result_tmp,\
                           'leaves':leaves,\
                          }]
                    all_result_dict[c_arg_dict['name']]= {'clusters':clusters,\
                           'Z':Z,\
                           'cluster_result_fcluster':cluster_result_tmp,\
                           'leaves':leaves,\
                          }
                    write_cluster_result_to_infoDict(all_result_tmp,all_info_selected)
                    eval_result = evaluate(arg,all_result_dict,all_info_selected,feature_clips_dict,\
                        gt_key = ['gt2_1411_black_two','gt1_1929_black_two'],\
                        exclude_unknown = True,\
                        gt_file = '/disk2/zexin/data/mice/behavior_data/Alphapose Ethogram Scoring Sheet.csv')
                    score = eval_result['Adjusted Rand index']
                    if score > score_select:
                        score_select = score
                        cluster_result = cluster_result_tmp
                        thred_select = threshold
                else:
                    print('only the following metricx are supported')
                    print('Adjusted Rand index\n Calinski-Harabasz Index \n Davies-Bouldin index ')
                    raise
                print('threshold %d: %s=%f, (score_select=%f,thred_select=%f)'%(threshold,c_arg_dict['evaluation_metric'],score,score_select,thred_select))
                c_arg_dict['thred'] = thred_select
        else:
            threshold = c_arg_dict['thred']
            cluster_result = hierarchy.fcluster(Z,threshold,'distance')

        clusters = [[] for i in range(np.max(cluster_result)+1)]
        for ii in range(len(cluster_result)):
            clusters[cluster_result[ii]].append(ii)

        clusters = sort_accor_leaves(clusters,leaves)
        # clusters = np.asarray(clusters,dtype = 'int16')
        cluster_result_fcluster = cluster_result
        print('number of clusters (selected):%d'%(np.max(cluster_result_fcluster)))  
    
        ### t sne 
        if arg.DR_method == 'tsne':
            n_components = 2 
            tsne = TSNE(n_components=n_components, init='pca', random_state=0)
            Y = tsne.fit_transform(feat_norm)
        elif arg.DR_method == 'umap':
            um = umap.UMAP(n_neighbors=5,
                          min_dist=0.3,
                          metric='correlation')
            Y = um.fit_transform(feat_norm)
        else:
            print('dimension reduction algorithm %s is not defined'%(arg.DR_method))
            raise
        
        all_result.append({'clusters':clusters,\
                           'Z':Z,\
                           'cluster_result_fcluster':cluster_result_fcluster,\
                           'leaves':leaves,\
                           'dimension_reduction_Y':Y
                          })
        all_result_dict[c_arg_dict['name']]= {'clusters':clusters,\
                           'Z':Z,\
                           'cluster_result_fcluster':cluster_result_fcluster,\
                           'leaves':leaves,\
                           'dimension_reduction_Y':Y
                          }

    return all_result, all_result_dict

def cluster_bak1(c_arg_dict,feature_clips):

    clip_num,fea_num = feature_clips.shape

    ## Perform hierarchical/agglomerative clustering.
    Z = linkage(feature_clips, method ='ward',metric='euclidean')

    # ## Plots the hierarchical clustering as a dendrogram
    dn = hierarchy.dendrogram(Z,orientation='top')
    leaves = np.asarray(dn['leaves'])

    ## Form flat clusters from the hierarchical clustering defined by the given linkage matrix.
    if isinstance(c_arg_dict['thred'],list):
        print('selecting fcluster threshold with %s metric:'%(c_arg_dict['evaluation_metric']))
        score_select = -100000
        for threshold in c_arg_dict['thred']:
            cluster_result_tmp = hierarchy.fcluster(Z,threshold,'distance')
            if c_arg_dict['evaluation_metric'] == 'Davies-Bouldin index':
                score = metrics.davies_bouldin_score(feature_clips, cluster_result_tmp)
                if -score > score_select:
                    score_select = -score
                    cluster_result = cluster_result_tmp
            elif c_arg_dict['evaluation_metric'] == 'Calinski-Harabasz Index':
                score = metrics.calinski_harabasz_score(feature_clips, cluster_result_tmp)
                if score > score_select:
                    score_select = score
                    cluster_result = cluster_result_tmp
            else:
                print('only Calinski-Harabasz Index and Davies-Bouldin index is supported')
                raise
            print('threshold %d: %s=%f, score_select=%f'%(threshold,c_arg_dict['evaluation_metric'],score,score_select))
    else:
        threshold = c_arg_dict['thred']
        cluster_result = hierarchy.fcluster(Z,threshold,'distance')

    clusters = [[] for i in range(np.max(cluster_result)+1)]
    for ii in range(len(cluster_result)):
        clusters[cluster_result[ii]].append(ii)

    clusters = sort_accor_leaves(clusters,leaves)
    # clusters = np.asarray(clusters,dtype = 'int16')

    return clusters,Z,cluster_result,leaves

def cluster_bak(arg,feature_clips):

    clip_num,fea_num = feature_clips.shape
    # feature_clips = np.reshape(feature_clips,(clip_num,-1))

    ## Perform hierarchical/agglomerative clustering.
    Z = linkage(feature_clips, method ='ward',metric='euclidean')
    # Z = linkage(feature_clips, method ='single',metric='euclidean')


    # ## Plots the hierarchical clustering as a dendrogram
    f = plt.figure(figsize = (15,5))
    g0 = plt.gca()
    dn = hierarchy.dendrogram(Z,
        above_threshold_color='y',
        orientation='top',
        ax = g0)
    # plt.tight_layout()
    leaves = np.asarray(dn['leaves'])
    # # print(leaves)
    # plt.savefig('temporal.png')
    # plt.show()
    plt.close()


    ## Form flat clusters from the hierarchical clustering defined by the given linkage matrix.
    threshold = arg.threshold
    cluster_result = hierarchy.fcluster(Z,threshold,'distance')

    clusters = [[] for i in range(np.max(cluster_result)+1)]
    for ii in range(len(cluster_result)):
        clusters[cluster_result[ii]].append(ii)

    clusters = sort_accor_leaves(clusters,leaves)
    # clusters = np.asarray(clusters,dtype = 'int16')

    return clusters,Z,cluster_result,leaves

def visualize(arg,clusters,pose_clips,info_clips,cont_clips,raw_feature_clips,Z, video_name_suffix=''):


    f = plt.figure(figsize = (15,5))
    g0 = plt.gca()
    dn = hierarchy.dendrogram(Z,
            above_threshold_color='y',
            orientation='top',
            ax = g0)
    plt.tight_layout()

    for cluster_ind in range(len(clusters)):

        visualize_one_cluster(arg,cluster_ind,clusters[cluster_ind],g0,f,pose_clips,info_clips,cont_clips,raw_feature_clips,video_name_suffix)

def left_right(pose_clips):

    # clip_num,frame_num,point_num,_ = pose_clips.shape
    clip_num,frame_num,point_num = pose_clips.shape[0],pose_clips.shape[1],pose_clips.shape[2]

    for clip_ind in range(clip_num):

        for frame_ind in range(frame_num):

            nose = pose_clips[clip_ind,frame_ind,0,:]
            ear1 = pose_clips[clip_ind,frame_ind,1,:]
            ear2 = pose_clips[clip_ind,frame_ind,2,:]
            vec1 = ear1-nose
            vec2 = ear2-nose

            _,phi1 = cart2pol(vec1[0],vec1[1])
            phi1 = math.degrees(phi1)

            _,phi2 = cart2pol(vec2[0],vec2[1])
            phi2 = math.degrees(phi2)

            if (phi1<phi2):
                if (phi2-phi1)<180:
                    flag = True
                else:
                    flag = False
            else:
                if (phi1-phi2)<180:
                    flag = False
                else:
                    flag = True

            if flag == True:
                temp = ear1.copy()
                pose_clips[clip_ind,frame_ind,1,:] = ear2
                pose_clips[clip_ind,frame_ind,2,:] = temp

    return pose_clips

def getPos(nose,tail,point):
    vec1 = tail - nose
    vec2 = point - nose
    _,phi1 = cart2pol(vec1[0],vec1[1])
    phi1 = math.degrees(phi1)
    _,phi2 = cart2pol(vec2[0],vec2[1])
    phi2 = math.degrees(phi2)

    if (phi1<phi2):
        if (phi2-phi1)<180:
            flag = True
        else:
            flag = False
    else:
        if (phi1-phi2)<180:
            flag = False
        else:
            flag = True

    if flag == True: 
        return 'right'
    else:
        return 'left'      

def correctLimbs(pose_clips):

    clip_num,frame_num,point_num = pose_clips.shape[0],pose_clips.shape[1],pose_clips.shape[2]

    if point_num != 8:
        return pose_clips

    for clip_ind in range(clip_num):
        for frame_ind in range(frame_num):

            nose = pose_clips[clip_ind,frame_ind,0,:]
            tail = pose_clips[clip_ind,frame_ind,3,:]
            left_forelimb = pose_clips[clip_ind,frame_ind,4,:]
            right_forelimb = pose_clips[clip_ind,frame_ind,5,:]
            left_backlimb = pose_clips[clip_ind,frame_ind,6,:]
            right_backlimb = pose_clips[clip_ind,frame_ind,7,:]

            if getPos(nose,tail,left_forelimb)=='right':
                pose_clips[clip_ind,frame_ind,4,:] = [0,0]
            if getPos(nose,tail,right_forelimb)=='left':
                pose_clips[clip_ind,frame_ind,5,:] = [0,0] 
            if getPos(nose,tail,left_backlimb)=='right':
                pose_clips[clip_ind,frame_ind,6,:] = [0,0]
            if getPos(nose,tail,right_backlimb)=='left':
                pose_clips[clip_ind,frame_ind,7,:] = [0,0] 

    return pose_clips

def sep_cluster(list_afterCluster,sep_name):
    last_len = len(list_afterCluster)
    new_cluster_list = []
    for i in range(last_len):
        tmp_dict = {}
        cluster_dict = list_afterCluster[i]
        last_cluster_name = cluster_dict['cluster_name']

        cluster_result_fcluster = cluster_dict['cluster_result_fcluster']
        for j in range(1, np.max(cluster_result_fcluster)+1):
            tmp_dict[last_cluster_name+'_'+sep_name+str(j)] = \
                {'pose_clips':[],\
                'info_clips':[],\
                'cont_clips':[],\
                'raw_feature_clips':[],\
                'feature_clips_sep':[[] for ii in range(len(cluster_dict['feature_clips_sep']))],\
                'ori_id':[],'cluster_name':last_cluster_name+'_'+sep_name+str(j)}
        
        for k in range(len(cluster_result_fcluster)):
            cluster_id = cluster_result_fcluster[k]
            dict_key = last_cluster_name+'_'+sep_name+str(cluster_id)
            tmp_dict[dict_key]['pose_clips'].append(cluster_dict['pose_clips'][k])
            tmp_dict[dict_key]['info_clips'].append(cluster_dict['info_clips'][k])
            tmp_dict[dict_key]['cont_clips'].append(cluster_dict['cont_clips'][k])
            tmp_dict[dict_key]['raw_feature_clips'].append(cluster_dict['raw_feature_clips'][k])
            tmp_dict[dict_key]['ori_id'].append(cluster_dict['ori_id'][k])
            for ii in range(len(cluster_dict['feature_clips_sep'])):
                tmp_dict[dict_key]['feature_clips_sep'][ii].append(cluster_dict['feature_clips_sep'][ii][k])

        for key in tmp_dict:
            new_cluster_list.append(tmp_dict[key])

    return new_cluster_list

def visualize_dict(arg,cluster_ids,name,clusters_dict,all_result,pose_clips,info_clips,cont_clips,raw_feature_clips_dict, video_name_suffix=''):


    for k in clusters_dict:
        all_clipID = clusters_dict[k]
        if(len(all_clipID)<5):
            continue

        all_clipID = sorted(all_clipID,key = lambda clip_id: np.nonzero(all_result[cluster_ids[0]]['leaves'] == clip_id)[0])

        out_video_path = 'results/cluster_{}_{}.mp4'.format(video_name_suffix,k)
        print(out_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, 9.0, (1400,800))

        # for clip_ind in tqdm(all_clipID[0:min(20,len(all_clipID))]):
        for clip_ind in tqdm(all_clipID):
            ############### draw dendrogram ###################
            for cluster_num_id in cluster_ids:
                Z = all_result[cluster_num_id]['Z']
                leaves = all_result[cluster_num_id]['leaves']
                f = plt.figure(figsize = (15,5))

                g0 = plt.gca()
                dn = hierarchy.dendrogram(Z,
                        above_threshold_color='y',
                        orientation='top',
                        ax = g0)
                plt.tight_layout()

                xlim  = g0.get_xlim()
                # g0.set_xlabel(name[cluster_num_id],fontsize=50)
                xlim = xlim[1]
                xx = np.squeeze(np.argwhere(leaves == clip_ind))
                xxx = xlim * xx/leaves.size

                ll = g0.plot([xxx,xxx],[0,arg.threshold],linewidth = 4,color = 'red')
                f.savefig('dendrogram_%d.png'%cluster_num_id)

                l = ll[0]
                l.remove()
                del l
                plt.close()


            ############### draw contour heatmap ###################
            fig,ax = plt.subplots()
            sns.heatmap(cont_clips[clip_ind,:,:])
            fig.savefig('cont_temp.png')
            plt.close()

            ############### draw feature ###################
            fig,axes = plt.subplots(3,2,figsize= (10,6))
            axes[0,0].plot(raw_feature_clips_dict[clip_ind]['body_length'])
            axes[0,0].set_ylim([80,200])
            axes[0,0].set_ylabel('Body Length')
            axes[0,1].plot(raw_feature_clips_dict[clip_ind]['head_length'])
            axes[0,1].set_ylim([0,80])
            axes[0,1].set_ylabel('Head Length')
            axes[1,0].plot(raw_feature_clips_dict[clip_ind]['head_body_angles'])
            axes[1,0].set_ylim([-1.7,1.7])
            axes[1,0].set_ylabel('Head Body Angle')
            axes[1,1].plot(raw_feature_clips_dict[clip_ind]['displace_x'])
            axes[1,1].set_ylim([-60,60])
            axes[1,1].set_ylabel('Displacement X')
            axes[2,0].plot(raw_feature_clips_dict[clip_ind]['displace_y'])
            axes[2,0].set_ylim([-40,40])
            axes[2,0].set_ylabel('Displacement Y')
            axes[2,1].plot(raw_feature_clips_dict[clip_ind]['displace_rho'])
            axes[2,1].set_ylim([-60,60])
            axes[2,1].set_ylabel('Displacement rho')
            fig.savefig('raw_temp.png')
            plt.close()

            pose = pose_clips[clip_ind][7]
            nose,tail = pose[0,:],pose[3,:]
            body = tail-nose
            rho,phi = cart2pol(body[0],body[1])
            angle = math.degrees(phi)

            for frame_ind in range(arg.clip_length):

                # print(clip_ind,frame_ind)
                img = cv2.imread(info_clips[clip_ind][frame_ind])
                try:
                    rows,cols,depth = img.shape
                except Exception as e:
                    print(info_clips[clip_ind][frame_ind])
                    print(clip_ind,frame_ind)
                    raise e
                

                M = np.float32([[1,0,640-nose[0]],[0,1,360-nose[1]]])
                tra = cv2.warpAffine(img,M,(cols,rows))


                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                rot = cv2.warpAffine(tra,M,(cols,rows))


                crop = rot[160:560,440:840].copy()

                cv2.putText(crop,'cluster_{}_clip_{}'.format(k,clip_ind),
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    arg.fontColor,
                    arg.lineType)


                cont = cv2.imread('cont_temp.png')
                cont = cv2.resize(cont,(400,400))
                img = cv2.hconcat((crop,cont))

                raw = cv2.imread('raw_temp.png')
                raw = cv2.resize(raw,(800,400))
                img = cv2.vconcat((img,raw))


                dend = cv2.imread('dendrogram_%d.png'%cluster_ids[0])
                dend = cv2.resize(dend,\
                    (600,800- (len(cluster_ids)-1)*int(800/len(cluster_ids))))
                cv2.putText(dend,name[cluster_ids[0]],
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (0,0,255),
                    arg.lineType)
                for cluster_num_id in cluster_ids[1:]:
                    dend1 = cv2.imread('dendrogram_%d.png'%(cluster_num_id))
                    dend1 = cv2.resize(dend1,(600,int(800/len(cluster_ids))))
                    cv2.putText(dend1,name[cluster_num_id],
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (0,0,255),
                    arg.lineType)

                    dend = cv2.vconcat((dend,dend1))
                img = cv2.hconcat((img,dend))


                out.write(img)


        out.release()

def visualize_allInOneVideo(arg,cluster_ids,name,clusters_dict,all_result,pose_clips,info_clips,cont_clips,raw_feature_clips_dict, feature_clips_dict, video_name_suffix=''):

    print('note: using zscore_all to norm all the feature in feature_clips_dict')
    feature_clips_dict_norm = {}
    for feature_key in feature_clips_dict:
        feature_clips_dict_norm[feature_key] = zscoreAll_norm(feature_clips_dict[feature_key])

    out_video_path = 'results/cluster_{}_all.mp4'.format(video_name_suffix)
    print(out_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 9.0, (1400,800))
    pre_clip_ind = all_result[cluster_ids[0]]['leaves'][0]
    for clip_ind_idx in tqdm(range(len(all_result[cluster_ids[0]]['leaves']))):
        # if(clip_ind_idx>3):
        #     break
        clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
        ############### draw dendrogram and  tsne  ###################
        for cluster_num_id in cluster_ids:
            ############### draw dendrogram ###################
            Z = all_result[cluster_num_id]['Z']
            leaves = all_result[cluster_num_id]['leaves']
            f = plt.figure(figsize = (15,5))

            g0 = plt.gca()
            dn = hierarchy.dendrogram(Z,
                    above_threshold_color='y',
                    orientation='top',
                    ax = g0)
            plt.tight_layout()

            xlim  = g0.get_xlim()
            # g0.set_xlabel(name[cluster_num_id],fontsize=50)
            xlim = xlim[1]
            xx = np.squeeze(np.argwhere(leaves == clip_ind))
            xxx = xlim * xx/leaves.size

            ll = g0.plot([xxx,xxx],[0,arg.threshold],linewidth = 4,color = 'red')
            f.savefig('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))

            l = ll[0]
            l.remove()
            del l
            plt.close()

            ############### draw  tsne ###################
            Y = all_result[cluster_num_id]['t_sne_Y']
            cluster_result_fcluster = all_result[cluster_num_id]['cluster_result_fcluster']
            color = [RGB_to_Hex([(50*c_id)%255,(80*c_id)%255,(110*c_id)%255]) for c_id in cluster_result_fcluster ]
            f = plt.figure(figsize = (15,5))
            plt.scatter(Y[:, 0], Y[:, 1],c=color, cmap=plt.cm.Spectral)
            plt.scatter([Y[clip_ind, 0]],[Y[clip_ind, 1]], marker = '*',color = 'red', s = 600 )
            # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
            # plt.show()
            f.savefig('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
            plt.close()

        ############### draw  feature distance ###################
        feature_name = []
        feature_distance1 = []
        feature_distance2 = []
        if clip_ind_idx+1 >= len(all_result[cluster_ids[0]]['leaves'])-1:
            next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
        else:
            next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx+1]
        for feature_key in feature_clips_dict:
            feature_name.append(feature_key)
            feature_distance1.append(
                square_sum(
                    feature_clips_dict_norm[feature_key][pre_clip_ind],
                    feature_clips_dict_norm[feature_key][clip_ind],
                    ))
            feature_distance2.append(
                square_sum(
                    feature_clips_dict_norm[feature_key][next_clip_ind],
                    feature_clips_dict_norm[feature_key][clip_ind],
                    ))
        pre_clip_ind = clip_ind

        f = plt.figure(figsize = (15,5))
        plt.barh(range(len(feature_distance1)), feature_distance1,color='rgb',tick_label=feature_name)
        f.savefig('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
        plt.close()
        f = plt.figure(figsize = (15,5))
        plt.barh(range(len(feature_distance2)), feature_distance2,color='rgb',tick_label=feature_name)
        f.savefig('./tmp_image/feature_distance2_%s_%d.png'%(video_name_suffix,cluster_num_id))
        plt.close()



        ############### draw contour heatmap ###################
        fig,ax = plt.subplots()
        sns.heatmap(cont_clips[clip_ind,:,:])
        fig.savefig('./tmp_image/cont_temp_%s.png'%(video_name_suffix))
        plt.close()

        ############### draw feature ###################
        fig,axes = plt.subplots(3,2,figsize= (10,6))
        axes[0,0].plot(raw_feature_clips_dict[clip_ind]['body_length'])
        axes[0,0].set_ylim([80,200])
        axes[0,0].set_ylabel('Body Length')
        axes[0,1].plot(raw_feature_clips_dict[clip_ind]['head_length'])
        axes[0,1].set_ylim([0,80])
        axes[0,1].set_ylabel('Head Length')
        axes[1,0].plot(raw_feature_clips_dict[clip_ind]['head_body_angles'])
        axes[1,0].set_ylim([-1.7,1.7])
        axes[1,0].set_ylabel('Head Body Angle')
        axes[1,1].plot(raw_feature_clips_dict[clip_ind]['displace_x'])
        axes[1,1].set_ylim([-60,60])
        axes[1,1].set_ylabel('Displacement X')
        axes[2,0].plot(raw_feature_clips_dict[clip_ind]['displace_y'])
        axes[2,0].set_ylim([-40,40])
        axes[2,0].set_ylabel('Displacement Y')
        axes[2,1].plot(raw_feature_clips_dict[clip_ind]['displace_rho'])
        axes[2,1].set_ylim([-60,60])
        axes[2,1].set_ylabel('Displacement rho')
        fig.savefig('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
        plt.close()

        pose = pose_clips[clip_ind][7]
        nose,tail = pose[0,:],pose[3,:]
        body = tail-nose
        rho,phi = cart2pol(body[0],body[1])
        angle = math.degrees(phi)

        for frame_ind in range(arg.clip_length):

            # print(clip_ind,frame_ind)
            img = cv2.imread(info_clips[clip_ind][frame_ind])
            try:
                rows,cols,depth = img.shape
            except Exception as e:
                print(info_clips[clip_ind][frame_ind])
                print(clip_ind,frame_ind)
                raise e
            

            M = np.float32([[1,0,640-nose[0]],[0,1,360-nose[1]]])
            tra = cv2.warpAffine(img,M,(cols,rows))


            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            rot = cv2.warpAffine(tra,M,(cols,rows))


            crop = rot[160:560,440:840].copy()

            cv2.putText(crop,'cluster_clip_{}'.format(clip_ind),
                arg.bottomLeftCornerOfText,
                arg.font,
                arg.fontScale,
                arg.fontColor,
                arg.lineType)


            # cont = cv2.imread('./tmp_image/cont_temp_%s.png'%(video_name_suffix))
            # cont = cv2.resize(cont,(400,400))
            # img = cv2.hconcat((crop,cont))

            # raw = cv2.imread('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
            # raw = cv2.resize(raw,(800,400))
            # img = cv2.vconcat((img,raw))

            raw = cv2.imread('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
            raw = cv2.resize(raw,(400,400))
            img = cv2.hconcat((crop,raw))


            f_dis1 = cv2.imread('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
            f_dis1 = cv2.resize(f_dis1,(800,400))
            img = cv2.vconcat((img,f_dis1))


            # num_pic_v = len(cluster_ids)*2+1
            # f_dis1 = cv2.imread('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
            # f_dis1 = cv2.resize(f_dis1,(300,800-(num_pic_v-1)*int(800/num_pic_v)))
            # f_dis2 = cv2.imread('./tmp_image/feature_distance2_%s_%d.png'%(video_name_suffix,cluster_num_id))
            # f_dis2 = cv2.resize(f_dis2,(300,800-(num_pic_v-1)*int(800/num_pic_v)))
            # dend_begin = cv2.hconcat((f_dis1,f_dis2))

            num_pic_v = len(cluster_ids)*2
            dend = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
            dend = cv2.resize(dend,(600,800-(num_pic_v-1)*int(800/num_pic_v)))
            cv2.putText(dend,name[cluster_ids[0]],
                arg.bottomLeftCornerOfText,
                arg.font,
                arg.fontScale,
                (0,0,255),
                arg.lineType)
            tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
            tsne = cv2.resize(tsne,(600,int(800/num_pic_v)))
            # dend = cv2.vconcat((dend_begin,dend))
            dend = cv2.vconcat((dend,tsne))
            for cluster_num_id in cluster_ids[1:]:
                dend1 = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))
                dend1 = cv2.resize(dend1,(600,int(800/num_pic_v)))
                cv2.putText(dend1,name[cluster_num_id],
                arg.bottomLeftCornerOfText,
                arg.font,
                arg.fontScale,
                (0,0,255),
                arg.lineType)
                dend = cv2.vconcat((dend,dend1))
                tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
                tsne = cv2.resize(tsne,(600,int(800/num_pic_v)))
                dend = cv2.vconcat((dend,tsne))
            img = cv2.hconcat((img,dend))


            out.write(img)


    out.release()

def sort_by_firstTwoFeature(cluster_merged_dict,cluster_ids,all_result,max_clips_num=60):
    cluster_tmp = []
    for k in cluster_merged_dict:
        all_clipID = cluster_merged_dict[k]
        all_clipID = sorted(all_clipID,key = lambda clip_id: np.nonzero(all_result[cluster_ids[0]]['leaves'] == clip_id)[0])
        cluster_tmp.append(all_clipID)
    cluster_tmp_sorted = sorted(cluster_tmp,key = lambda clip_cluster: np.nonzero(all_result[cluster_ids[0]]['leaves'] == clip_cluster[0])[0])

    all_clipID_sorted = []
    all_clipID_sorted_sep = []

    for i in range(len(cluster_tmp_sorted)):
        if (len(cluster_ids)>1):
            list_toAdd = sorted(cluster_tmp_sorted[i],key = lambda clip_id: np.nonzero(all_result[cluster_ids[1]]['leaves'] == clip_id)[0])
        else:
            list_toAdd = cluster_tmp_sorted[i]
        all_clipID_sorted += list_toAdd[:min(max_clips_num,len(list_toAdd))]
        all_clipID_sorted_sep.append(list_toAdd[:min(max_clips_num,len(list_toAdd))])
    return all_clipID_sorted, all_clipID_sorted_sep

def visualize_inOneVideo_sort(arg,\
    cluster_ids,cluster_ids_name,cluster_merged_dict,
    all_result, \
    pose_clips,info_clips,cont_clips,raw_feature_clips_dict, feature_clips_dict, \
    video_name_suffix='',gen_video_folder='./results/',max_clips_num=60):

    print('note: using zscore_all to norm all the feature in feature_clips_dict')
    feature_clips_dict_norm = {}
    for feature_key in feature_clips_dict:
        feature_clips_dict_norm[feature_key] = zscoreAll_norm(feature_clips_dict[feature_key])

    out_video_path = '{}/cluster_{}_all.mp4'.format(gen_video_folder,video_name_suffix)
    print(out_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 9.0, (1400,800))


    all_clipID_sorted,all_clipID_sorted_sep = sort_by_firstTwoFeature(cluster_merged_dict,cluster_ids,all_result,max_clips_num=max_clips_num)
    

    # pre_clip_ind = all_result[cluster_ids[0]]['leaves'][0]
    # for clip_ind_idx in tqdm(range(len(all_result[cluster_ids[0]]['leaves']))):
    pre_clip_ind = all_clipID_sorted[0]
    for clip_ind_idx in tqdm(range(len(all_clipID_sorted))):

        # if(clip_ind_idx>5):
        #     break
        # clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
        clip_ind = all_clipID_sorted[clip_ind_idx]
        ############### draw dendrogram and  tsne  ###################
        for cluster_num_id in cluster_ids:
            ############### draw dendrogram ###################
            Z = all_result[cluster_num_id]['Z']
            leaves = all_result[cluster_num_id]['leaves']
            f = plt.figure(figsize = (15,5))

            g0 = plt.gca()
            dn = hierarchy.dendrogram(Z,
                    above_threshold_color='y',
                    orientation='top',
                    ax = g0)
            plt.tight_layout()

            xlim  = g0.get_xlim()
            # g0.set_xlabel(cluster_ids_name[cluster_num_id],fontsize=50)
            xlim = xlim[1]
            xx = np.squeeze(np.argwhere(leaves == clip_ind))
            xxx = xlim * xx/leaves.size

            ll = g0.plot([xxx,xxx],[0,arg.threshold],linewidth = 4,color = 'red')
            f.savefig('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))

            l = ll[0]
            l.remove()
            del l
            plt.close()

            ############### draw  tsne ###################
            Y = all_result[cluster_num_id]['t_sne_Y']
            cluster_result_fcluster = all_result[cluster_num_id]['cluster_result_fcluster']
            color = [RGB_to_Hex([(50*c_id)%255,(80*c_id)%255,(110*c_id)%255]) for c_id in cluster_result_fcluster ]
            f = plt.figure(figsize = (15,5))
            plt.scatter(Y[:, 0], Y[:, 1],c=color, cmap=plt.cm.Spectral)
            plt.scatter([Y[clip_ind, 0]],[Y[clip_ind, 1]], marker = '*',color = 'red', s = 600 )
            # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
            # plt.show()
            f.savefig('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
            plt.close()

        ############### draw  feature distance ###################
        feature_name = []
        feature_distance1 = []
        feature_distance2 = []
        if clip_ind_idx+1 >= len(all_result[cluster_ids[0]]['leaves'])-1:
            next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
        else:
            next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx+1]
        for feature_key in feature_clips_dict:
            feature_name.append(feature_key)
            feature_distance1.append(
                square_sum(
                    feature_clips_dict_norm[feature_key][pre_clip_ind],
                    feature_clips_dict_norm[feature_key][clip_ind],
                    ))
            feature_distance2.append(
                square_sum(
                    feature_clips_dict_norm[feature_key][next_clip_ind],
                    feature_clips_dict_norm[feature_key][clip_ind],
                    ))
        pre_clip_ind = clip_ind
        feature_distance1 = [1/(f_dist+0.001) for f_dist in feature_distance1]
        feature_distance2 = [1/(f_dist+0.001) for f_dist in feature_distance2]

        f = plt.figure(figsize = (15,5))
        plt.barh(range(len(feature_distance1)), feature_distance1,color='rgb',tick_label=feature_name)
        plt.xlabel('importance of feature (1/distance)')
        f.savefig('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
        plt.close()
        f = plt.figure(figsize = (15,5))
        plt.barh(range(len(feature_distance2)), feature_distance2,color='rgb',tick_label=feature_name)
        plt.xlabel('importance of feature')
        f.savefig('./tmp_image/feature_distance2_%s_%d.png'%(video_name_suffix,cluster_num_id))
        plt.close()



        ############### draw contour heatmap ###################
        fig,ax = plt.subplots()
        sns.heatmap(cont_clips[clip_ind,:,:])
        fig.savefig('./tmp_image/cont_temp_%s.png'%(video_name_suffix))
        plt.close()

        ############### draw feature ###################
        fig,axes = plt.subplots(3,2,figsize= (10,6))
        axes[0,0].plot(raw_feature_clips_dict[clip_ind]['body_length'])
        axes[0,0].set_ylim([80,200])
        axes[0,0].set_ylabel('Body Length')
        axes[0,1].plot(raw_feature_clips_dict[clip_ind]['head_length'])
        axes[0,1].set_ylim([0,80])
        axes[0,1].set_ylabel('Head Length')
        axes[1,0].plot(raw_feature_clips_dict[clip_ind]['head_body_angles'])
        axes[1,0].set_ylim([-1.7,1.7])
        axes[1,0].set_ylabel('Head Body Angle')
        axes[1,1].plot(raw_feature_clips_dict[clip_ind]['displace_x'])
        axes[1,1].set_ylim([-60,60])
        axes[1,1].set_ylabel('Displacement X')
        axes[2,0].plot(raw_feature_clips_dict[clip_ind]['displace_y'])
        axes[2,0].set_ylim([-40,40])
        axes[2,0].set_ylabel('Displacement Y')
        axes[2,1].plot(raw_feature_clips_dict[clip_ind]['displace_rho'])
        axes[2,1].set_ylim([-60,60])
        axes[2,1].set_ylabel('Displacement rho')
        fig.savefig('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
        plt.close()

        ############### get cluster ids ###################
        cluster_ids_s = ''
        for cluster_num_id in cluster_ids:
            cluster_ids_s += str(all_result[cluster_num_id]['cluster_result_fcluster'][clip_ind])+'_'
        cluster_ids_s = cluster_ids_s[:-1]


        pose = pose_clips[clip_ind][7]
        nose,tail = pose[0,:],pose[3,:]
        body = tail-nose
        rho,phi = cart2pol(body[0],body[1])
        angle = math.degrees(phi)

        for frame_ind in range(arg.clip_length):

            # print(clip_ind,frame_ind)
            img = cv2.imread(info_clips[clip_ind][frame_ind])
            try:
                rows,cols,depth = img.shape
            except Exception as e:
                print(info_clips[clip_ind][frame_ind])
                print(clip_ind,frame_ind)
                raise e
            

            M = np.float32([[1,0,640-nose[0]],[0,1,360-nose[1]]])
            tra = cv2.warpAffine(img,M,(cols,rows))


            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            rot = cv2.warpAffine(tra,M,(cols,rows))


            crop = rot[160:560,440:840].copy()

            cv2.putText(crop,'cluster_{}_clip_{}'.format(cluster_ids_s,clip_ind),
                arg.bottomLeftCornerOfText,
                arg.font,
                arg.fontScale,
                arg.fontColor,
                arg.lineType)


            # cont = cv2.imread('./tmp_image/cont_temp_%s.png'%(video_name_suffix))
            # cont = cv2.resize(cont,(400,400))
            # img = cv2.hconcat((crop,cont))

            # raw = cv2.imread('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
            # raw = cv2.resize(raw,(800,400))
            # img = cv2.vconcat((img,raw))

            raw = cv2.imread('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
            raw = cv2.resize(raw,(400,400))
            img = cv2.hconcat((crop,raw))

            f_dis1 = cv2.imread('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
            f_dis1 = cv2.resize(f_dis1,(800,400))
            img = cv2.vconcat((img,f_dis1))


            # num_pic_v = len(cluster_ids)*2+1
            # f_dis1 = cv2.imread('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
            # f_dis1 = cv2.resize(f_dis1,(300,800-(num_pic_v-1)*int(800/num_pic_v)))
            # f_dis2 = cv2.imread('./tmp_image/feature_distance2_%s_%d.png'%(video_name_suffix,cluster_num_id))
            # f_dis2 = cv2.resize(f_dis2,(300,800-(num_pic_v-1)*int(800/num_pic_v)))
            # dend_begin = cv2.hconcat((f_dis1,f_dis2))

            num_pic_v = len(cluster_ids)*2
            dend = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
            dend = cv2.resize(dend,(600,800-(num_pic_v-1)*int(800/num_pic_v)))
            cv2.putText(dend,cluster_ids_name[cluster_ids[0]],
                arg.bottomLeftCornerOfText,
                arg.font,
                arg.fontScale,
                (0,0,255),
                arg.lineType)
            tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
            tsne = cv2.resize(tsne,(600,int(800/num_pic_v)))
            # dend = cv2.vconcat((dend_begin,dend))
            dend = cv2.vconcat((dend,tsne))
            for cluster_num_id in cluster_ids[1:]:
                dend1 = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))
                dend1 = cv2.resize(dend1,(600,int(800/num_pic_v)))
                cv2.putText(dend1,cluster_ids_name[cluster_num_id],
                arg.bottomLeftCornerOfText,
                arg.font,
                arg.fontScale,
                (0,0,255),
                arg.lineType)
                dend = cv2.vconcat((dend,dend1))
                tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
                tsne = cv2.resize(tsne,(600,int(800/num_pic_v)))
                dend = cv2.vconcat((dend,tsne))
            img = cv2.hconcat((img,dend))


            out.write(img)


    out.release()

def visualize_inSepVideo(arg,\
    cluster_ids,cluster_ids_name,cluster_merged_dict,
    all_result, \
    pose_clips,info_clips,cont_clips,raw_feature_clips_dict, feature_clips_dict, \
    video_name_suffix='',gen_video_folder='./results/',max_clips_num=60):

    print('note: using zscore_all to norm all the feature in feature_clips_dict')
    feature_clips_dict_norm = {}
    for feature_key in feature_clips_dict:
        feature_clips_dict_norm[feature_key] = zscoreAll_norm(feature_clips_dict[feature_key])

    out_video_path = '{}/cluster_{}_all.mp4'.format(gen_video_folder,video_name_suffix)
    print(out_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_all = cv2.VideoWriter(out_video_path, fourcc, 9.0, (1400,800))


    _,all_clipID_sorted_sep = sort_by_firstTwoFeature(cluster_merged_dict,cluster_ids,all_result,max_clips_num=max_clips_num)
    

    # pre_clip_ind = all_result[cluster_ids[0]]['leaves'][0]
    # for clip_ind_idx in tqdm(range(len(all_result[cluster_ids[0]]['leaves']))):
    for iii in range(len(all_clipID_sorted_sep)):
        all_clipID_sorted = all_clipID_sorted_sep[iii]
        ############### get cluster ids ###################
        cluster_ids_s = ''
        for cluster_num_id in cluster_ids:
            cluster_ids_s += str(all_result[cluster_num_id]['cluster_result_fcluster'][all_clipID_sorted[0]])+'_'
        cluster_ids_s = cluster_ids_s[:-1]

        out_video_path = '{}/cluster_{}_{}.mp4'.format(gen_video_folder,video_name_suffix,cluster_ids_s)
        print(out_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, 9.0, (1400,800))

        pre_clip_ind = all_clipID_sorted[0]
        for clip_ind_idx in tqdm(range(len(all_clipID_sorted))):
            

            # if(clip_ind_idx>5):
            #     break
            # clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
            clip_ind = all_clipID_sorted[clip_ind_idx]
            ############### draw dendrogram and  tsne  ###################
            for cluster_num_id in cluster_ids:
                ############### draw dendrogram ###################
                Z = all_result[cluster_num_id]['Z']
                leaves = all_result[cluster_num_id]['leaves']
                f = plt.figure(figsize = (15,5))

                g0 = plt.gca()
                dn = hierarchy.dendrogram(Z,
                        above_threshold_color='y',
                        orientation='top',
                        ax = g0)
                plt.tight_layout()

                xlim  = g0.get_xlim()
                # g0.set_xlabel(cluster_ids_name[cluster_num_id],fontsize=50)
                xlim = xlim[1]
                xx = np.squeeze(np.argwhere(leaves == clip_ind))
                xxx = xlim * xx/leaves.size

                ll = g0.plot([xxx,xxx],[0,arg.threshold],linewidth = 4,color = 'red')
                f.savefig('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))

                l = ll[0]
                l.remove()
                del l
                plt.close()

                ############### draw  tsne ###################
                Y = all_result[cluster_num_id]['t_sne_Y']
                cluster_result_fcluster = all_result[cluster_num_id]['cluster_result_fcluster']
                color = [RGB_to_Hex([(50*c_id)%255,(80*c_id)%255,(110*c_id)%255]) for c_id in cluster_result_fcluster ]
                f = plt.figure(figsize = (15,5))
                plt.scatter(Y[:, 0], Y[:, 1],c=color, cmap=plt.cm.Spectral)
                plt.scatter([Y[clip_ind, 0]],[Y[clip_ind, 1]], marker = '*',color = 'red', s = 600 )
                # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
                # plt.show()
                f.savefig('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
                plt.close()

            ############### draw  feature distance ###################
            feature_name = []
            feature_distance1 = []
            feature_distance2 = []
            if clip_ind_idx+1 >= len(all_result[cluster_ids[0]]['leaves'])-1:
                next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
            else:
                next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx+1]
            for feature_key in feature_clips_dict:
                feature_name.append(feature_key)
                feature_distance1.append(
                    square_sum(
                        feature_clips_dict_norm[feature_key][pre_clip_ind],
                        feature_clips_dict_norm[feature_key][clip_ind],
                        ))
                feature_distance2.append(
                    square_sum(
                        feature_clips_dict_norm[feature_key][next_clip_ind],
                        feature_clips_dict_norm[feature_key][clip_ind],
                        ))
            pre_clip_ind = clip_ind
            feature_distance1 = [1/(f_dist+0.001) for f_dist in feature_distance1]
            feature_distance2 = [1/(f_dist+0.001) for f_dist in feature_distance2]

            f = plt.figure(figsize = (15,5))
            plt.barh(range(len(feature_distance1)), feature_distance1,color='rgb',tick_label=feature_name)
            plt.xlabel('importance of feature (1/distance)')
            f.savefig('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
            plt.close()
            f = plt.figure(figsize = (15,5))
            plt.barh(range(len(feature_distance2)), feature_distance2,color='rgb',tick_label=feature_name)
            plt.xlabel('importance of feature')
            f.savefig('./tmp_image/feature_distance2_%s_%d.png'%(video_name_suffix,cluster_num_id))
            plt.close()



            ############### draw contour heatmap ###################
            fig,ax = plt.subplots()
            sns.heatmap(cont_clips[clip_ind,:,:])
            fig.savefig('./tmp_image/cont_temp_%s.png'%(video_name_suffix))
            plt.close()

            ############### draw feature ###################
            fig,axes = plt.subplots(3,2,figsize= (10,6))
            axes[0,0].plot(raw_feature_clips_dict[clip_ind]['body_length'])
            axes[0,0].set_ylim([80,200])
            axes[0,0].set_ylabel('Body Length')
            axes[0,1].plot(raw_feature_clips_dict[clip_ind]['head_length'])
            axes[0,1].set_ylim([0,80])
            axes[0,1].set_ylabel('Head Length')
            axes[1,0].plot(raw_feature_clips_dict[clip_ind]['head_body_angles'])
            axes[1,0].set_ylim([-1.7,1.7])
            axes[1,0].set_ylabel('Head Body Angle')
            axes[1,1].plot(raw_feature_clips_dict[clip_ind]['displace_x'])
            axes[1,1].set_ylim([-60,60])
            axes[1,1].set_ylabel('Displacement X')
            axes[2,0].plot(raw_feature_clips_dict[clip_ind]['displace_y'])
            axes[2,0].set_ylim([-40,40])
            axes[2,0].set_ylabel('Displacement Y')
            axes[2,1].plot(raw_feature_clips_dict[clip_ind]['displace_rho'])
            axes[2,1].set_ylim([-60,60])
            axes[2,1].set_ylabel('Displacement rho')
            fig.savefig('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
            plt.close()

            


            pose = pose_clips[clip_ind][7]
            nose,tail = pose[0,:],pose[3,:]
            body = tail-nose
            rho,phi = cart2pol(body[0],body[1])
            angle = math.degrees(phi)

            for frame_ind in range(arg.clip_length):

                # print(clip_ind,frame_ind)
                img = cv2.imread(info_clips[clip_ind][frame_ind])
                try:
                    rows,cols,depth = img.shape
                except Exception as e:
                    print('info of image that can not be read:')
                    print(info_clips[clip_ind][frame_ind])
                    print(clip_ind,frame_ind)
                    raise e
                

                M = np.float32([[1,0,640-nose[0]],[0,1,360-nose[1]]])
                tra = cv2.warpAffine(img,M,(cols,rows))


                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                rot = cv2.warpAffine(tra,M,(cols,rows))


                crop = rot[160:560,440:840].copy()

                cv2.putText(crop,'cluster_{}_clip_{}'.format(cluster_ids_s,clip_ind),
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    arg.fontColor,
                    arg.lineType)


                # cont = cv2.imread('./tmp_image/cont_temp_%s.png'%(video_name_suffix))
                # cont = cv2.resize(cont,(400,400))
                # img = cv2.hconcat((crop,cont))

                # raw = cv2.imread('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
                # raw = cv2.resize(raw,(800,400))
                # img = cv2.vconcat((img,raw))

                raw = cv2.imread('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
                raw = cv2.resize(raw,(400,400))
                img = cv2.hconcat((crop,raw))

                f_dis1 = cv2.imread('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
                f_dis1 = cv2.resize(f_dis1,(800,400))
                img = cv2.vconcat((img,f_dis1))


                # num_pic_v = len(cluster_ids)*2+1
                # f_dis1 = cv2.imread('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
                # f_dis1 = cv2.resize(f_dis1,(300,800-(num_pic_v-1)*int(800/num_pic_v)))
                # f_dis2 = cv2.imread('./tmp_image/feature_distance2_%s_%d.png'%(video_name_suffix,cluster_num_id))
                # f_dis2 = cv2.resize(f_dis2,(300,800-(num_pic_v-1)*int(800/num_pic_v)))
                # dend_begin = cv2.hconcat((f_dis1,f_dis2))

                num_pic_v = len(cluster_ids)*2
                dend = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
                dend = cv2.resize(dend,(600,800-(num_pic_v-1)*int(800/num_pic_v)))
                cv2.putText(dend,cluster_ids_name[cluster_ids[0]],
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (0,0,255),
                    arg.lineType)
                tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
                tsne = cv2.resize(tsne,(600,int(800/num_pic_v)))
                # dend = cv2.vconcat((dend_begin,dend))
                dend = cv2.vconcat((dend,tsne))
                for cluster_num_id in cluster_ids[1:]:
                    dend1 = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))
                    dend1 = cv2.resize(dend1,(600,int(800/num_pic_v)))
                    cv2.putText(dend1,cluster_ids_name[cluster_num_id],
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (0,0,255),
                    arg.lineType)
                    dend = cv2.vconcat((dend,dend1))
                    tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
                    tsne = cv2.resize(tsne,(600,int(800/num_pic_v)))
                    dend = cv2.vconcat((dend,tsne))
                img = cv2.hconcat((img,dend))


                out.write(img)
                out_all.write(img)
        out.release()


    out_all.release()

def write_cluster_result_to_infoDict(all_result,all_info_selected,Z_index_in_result_dict=0):
    # write inorder traversal of the tree to the clip dict
    Z = all_result[Z_index_in_result_dict]['Z']
    dn = hierarchy.dendrogram(Z,orientation='top')
    leaves = np.asarray(dn['leaves'])
    for i in range(len(leaves)):
        cilp_idx = leaves[i]
        all_info_selected['info_clips'][cilp_idx]['inorder_traversal_id'] = i

    for i in range(len(all_info_selected['info_clips'])):
        # all_info_selected['info_clips'][i]['clip_id'] = float(i)
        all_info_selected['info_clips'][i]['cluster_id'] = float(all_result[0]['cluster_result_fcluster'][i])

def process_Z(Z,all_info_selected):
    Z_return = []
    cluster_names = [str(int(all_info_selected['info_clips'][i]['cluster_id'])) for i in range(len(all_info_selected['info_clips']))]
    for i in range(Z.shape[0]):
        # print(int(Z[i,0]),int(Z[i,1]))
        if cluster_names[int(Z[i,0])] == cluster_names[int(Z[i,1])]:
            current_cluster_name = cluster_names[int(Z[i,0])]
        else:
            a = {ci for ci in cluster_names[int(Z[i,0])].split('_')}
            b = {ci for ci in cluster_names[int(Z[i,1])].split('_')}
            c = list(a.union(b))
            current_cluster_name = c[0]
            for j in range(1,len(c)):
                current_cluster_name = current_cluster_name + '_' + c[j]
        cluster_names.append(current_cluster_name)
        z_list = Z[i].tolist()
        z_list.append(current_cluster_name)
        Z_return.append(z_list)
    return Z_return


def save_Z_and_clips(arg,all_result_dict,all_info_selected):
    frames_path_clips = all_info_selected['frames_path_clips']
    if arg.dataPath_for_UI == '':
        return

    print('saving data for UI...')
    print('(you can set dataPath_for_UI=\'\' in setting.py to disable this saving.)')
    mkdir_p(arg.dataPath_for_UI)
    ## write Z

    print('writing Z...')
    for k in tqdm(all_result_dict):
        json_file_name = 'Z_'+k+'.json'
        with open(arg.dataPath_for_UI+'/'+json_file_name, 'w') as outfile:
            json.dump(process_Z(all_result_dict[k]['Z'],all_info_selected), outfile)

    ## write clips information
    print('writing clips information to %s...'%(arg.dataPath_for_UI+'/clips_info.json'))
    with open(arg.dataPath_for_UI+'/clips_info.json','w') as outfile:
        json.dump(all_info_selected['info_clips'],outfile)        

    ## write cilps
    print('writing clips...')
    videos_root = arg.dataPath_for_UI+'all_clips/'
    mkdir_p(videos_root)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    

    for clip_ind in tqdm(range(len(frames_path_clips))):
        video_path = videos_root+'/%d.mp4'%clip_ind
        # video_path = videos_root+'/%d.avi'%clip_ind
        print(video_path)
        img_tmp = cv2.imread(frames_path_clips[clip_ind][0])
        total_y,total_x,depth = img_tmp.shape
        out = cv2.VideoWriter(video_path, fourcc, 9.0, (total_x,total_y))
        for frame_ind in range(arg.clip_length):
            img = cv2.imread(frames_path_clips[clip_ind][frame_ind])
            try:
                # print(out)
                rows,cols,depth = img.shape
                width, height = img.shape[1],img.shape[0]
                out.write(img)
            except Exception as e:
                print('info of image that can not be read:')
                print(info_clips[clip_ind][frame_ind])
                print(clip_ind,frame_ind)
                raise e
        out.release()

def visualize_inSepVideo_twoMice(arg,\
    cluster_ids,cluster_ids_name,cluster_merged_dict,
    all_result, \
    pose_clips,poseTheOther_clips,info_clips,cont_clips,raw_feature_clips_dict, feature_clips_dict, \
    video_name_suffix='',gen_video_folder='./results/',max_clips_num=60):

    print('note: using zscore_all to norm all the feature in feature_clips_dict')
    feature_clips_dict_norm = {}
    for feature_key in feature_clips_dict:
        feature_clips_dict_norm[feature_key] = zscoreAll_norm(feature_clips_dict[feature_key])

    mkdir_p('./tmp_image')
    mkdir_p(gen_video_folder)
    out_video_path = '{}/cluster_{}_all.mp4'.format(gen_video_folder,video_name_suffix)
    print(out_video_path)

    
    ori_crop_halfWidth = 400
    ori_crop_halfHeight = 400
    crop_x = crop_y = final_crop_size = 400
    ori_img_y = crop_y*2
    ori_img_x = int(crop_y*2*1.43)
    raw_x = int((ori_img_x+crop_x)* 1/3 )
    feature_x = int( ori_img_x+crop_x - raw_x )
    raw_y = feature_y = 700
    dendrogram_x = 800
    dendrogram_y = ori_img_y + raw_y
    total_x = ori_img_x + crop_x + dendrogram_x
    total_y = dendrogram_y
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_all = cv2.VideoWriter(out_video_path, fourcc, 9.0, (total_x,total_y))

    _,all_clipID_sorted_sep = sort_by_firstTwoFeature(cluster_merged_dict,cluster_ids,all_result,max_clips_num=max_clips_num)

    ############### draw dendrogram  ###################
    dendrogram_info = [{} for i in range(max(cluster_ids)+1)]
    for cluster_num_id in cluster_ids:
        Z = all_result[cluster_num_id]['Z']
        leaves = all_result[cluster_num_id]['leaves']
        cluster_result_fcluster=all_result[cluster_num_id]['cluster_result_fcluster']
        color = [RGB_to_Hex([(53*c_id)%255,(83*c_id)%255,(113*c_id)%255]) for c_id in cluster_result_fcluster ]
        for i in range(Z.shape[0]):
            if Z[i,2]>arg.cluster_arg[cluster_num_id]['thred']:
                color.append('y')
            else:
                color.append(color[int(Z[i,0])])
        f = plt.figure(figsize = (15,5))
        g0 = plt.gca()
        dn = hierarchy.dendrogram(Z,
                above_threshold_color='y',
                link_color_func=lambda x: color[x],
                orientation='top',
                ax = g0)
        plt.tight_layout()
        xlim  = g0.get_xlim()
        xlim = xlim[1]
        dendrogram_info[cluster_num_id] = {'f':f, \
                                           'g0':g0,\
                                           'xlim':xlim }

    # for clip_ind_idx in tqdm(range(len(all_result[cluster_ids[0]]['leaves']))):
    for iii in range(len(all_clipID_sorted_sep)):
        all_clipID_sorted = all_clipID_sorted_sep[iii]
        ############### get cluster ids ###################
        cluster_ids_s = ''
        for cluster_num_id in cluster_ids:
            cluster_ids_s += str(all_result[cluster_num_id]['cluster_result_fcluster'][all_clipID_sorted[0]])+'_'
        cluster_ids_s = cluster_ids_s[:-1]

        out_video_path = '{}/cluster_{}_{}.mp4'.format(gen_video_folder,video_name_suffix,cluster_ids_s)
        print('generating %d/%d'%(iii,len(all_clipID_sorted_sep)))
        print(out_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, arg.video_fps, (total_x,total_y))

        pre_clip_ind = all_clipID_sorted[0]
        for clip_ind_idx in tqdm(range(len(all_clipID_sorted))):
            # if(clip_ind_idx>3):
            #     break
            # clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
            clip_ind = all_clipID_sorted[clip_ind_idx]
            ############### draw dendrogram and  tsne  ###################
            for cluster_num_id in cluster_ids:
                ############### draw dendrogram ###################
                t0 = time.time()
                leaves = all_result[cluster_num_id]['leaves']
                xlim = dendrogram_info[cluster_num_id]['xlim']
                g0 = dendrogram_info[cluster_num_id]['g0']
                f = dendrogram_info[cluster_num_id]['f']
                t01 = time.time()
                xx = np.squeeze(np.argwhere(leaves == clip_ind))
                xxx = xlim * xx/leaves.size + xlim/leaves.size/2
                ll = g0.plot([xxx,xxx],[0,arg.cluster_arg[cluster_num_id]['thred']],linewidth = 4,color = 'red')
                f.savefig('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))
                l = ll.pop(0)
                l.remove()
                del l

                ############### draw  tsne ###################
                t1 = time.time()
                Y = all_result[cluster_num_id]['dimension_reduction_Y']
                cluster_result_fcluster = all_result[cluster_num_id]['cluster_result_fcluster']
                color = [RGB_to_Hex([(50*c_id)%255,(80*c_id)%255,(110*c_id)%255]) for c_id in cluster_result_fcluster ]
                f = plt.figure(figsize = (15,5))
                plt.scatter(Y[:, 0], Y[:, 1],c=color, cmap=plt.cm.Spectral)
                plt.scatter([Y[clip_ind, 0]],[Y[clip_ind, 1]], marker = '*',color = 'red', s = 600 )
                # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
                # plt.show()
                f.savefig('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
                plt.close()

            ############### draw  feature distance ###################
            t2 = time.time()
            feature_name = []
            feature_distance1 = []
            feature_distance2 = []
            if clip_ind_idx+1 >= len(all_result[cluster_ids[0]]['leaves'])-1:
                next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx]
            else:
                next_clip_ind = all_result[cluster_ids[0]]['leaves'][clip_ind_idx+1]
            for feature_key in feature_clips_dict:
                feature_name.append(feature_key)
                feature_distance1.append(
                    square_sum(
                        feature_clips_dict_norm[feature_key][pre_clip_ind],
                        feature_clips_dict_norm[feature_key][clip_ind],
                        ))
                feature_distance2.append(
                    square_sum(
                        feature_clips_dict_norm[feature_key][next_clip_ind],
                        feature_clips_dict_norm[feature_key][clip_ind],
                        ))
            pre_clip_ind = clip_ind
            feature_distance1 = [1/(f_dist+0.001) for f_dist in feature_distance1]
            feature_distance2 = [1/(f_dist+0.001) for f_dist in feature_distance2]

            f = plt.figure(figsize = (10,5))
            plt.barh(range(len(feature_distance1)), feature_distance1,color='rgb',tick_label=feature_name)
            plt.xlabel('importance of feature (1/distance)', fontsize=14)
            f.savefig('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
            plt.close()
            f = plt.figure(figsize = (15,5))
            plt.barh(range(len(feature_distance2)), feature_distance2,color='rgb',tick_label=feature_name)
            plt.xlabel('importance of feature')
            f.savefig('./tmp_image/feature_distance2_%s_%d.png'%(video_name_suffix,cluster_num_id))
            plt.close()



            ############### draw contour heatmap ###################
            t3 = time.time()
            fig,ax = plt.subplots()
            sns.heatmap(cont_clips[clip_ind,:,:])
            fig.savefig('./tmp_image/cont_temp_%s.png'%(video_name_suffix))
            plt.close()

            ############### draw feature ###################
            fontsize = 16
            t4 = time.time()
            fig,axes = plt.subplots(3,2,figsize= (10,6))
            axes[0,0].plot(raw_feature_clips_dict[clip_ind]['body_length'])
            axes[0,0].set_ylim([80,200])
            axes[0,0].set_ylabel('Body Length', fontsize=fontsize)
            axes[0,1].plot(raw_feature_clips_dict[clip_ind]['head_length'])
            axes[0,1].set_ylim([0,80])
            axes[0,1].set_ylabel('Head Length', fontsize=fontsize)
            axes[1,0].plot(raw_feature_clips_dict[clip_ind]['head_body_angles'])
            axes[1,0].set_ylim([-1.7,1.7])
            axes[1,0].set_ylabel('Head Body Angle', fontsize=fontsize)
            axes[1,1].plot(raw_feature_clips_dict[clip_ind]['displace_x'])
            axes[1,1].set_ylim([-60,60])
            axes[1,1].set_ylabel('Displacement X', fontsize=fontsize)
            axes[2,0].plot(raw_feature_clips_dict[clip_ind]['displace_y'])
            axes[2,0].set_ylim([-40,40])
            axes[2,0].set_ylabel('Displacement Y', fontsize=fontsize)
            axes[2,1].plot(raw_feature_clips_dict[clip_ind]['displace_rho'])
            axes[2,1].set_ylim([-60,60])
            axes[2,1].set_ylabel('Displacement rho', fontsize=fontsize)
            fig.savefig('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
            plt.close()


            ############### draw frame ###################
            t5 = time.time()
            pose = pose_clips[clip_ind][7]
            nose,tail = pose[0,:],pose[3,:]
            body = tail-nose
            rho,phi = cart2pol(body[0],body[1])
            angle = math.degrees(phi)

            pose_ref = poseTheOther_clips[clip_ind][7]
            nose_ref,tail_ref = pose_ref[0,:],pose_ref[3,:]
            body_ref = tail_ref-nose_ref
            rho_ref,phi_ref = cart2pol(body_ref[0],body_ref[1])
            angle_ref = math.degrees(phi_ref)

            for frame_ind in range(arg.clip_length):
                img = cv2.imread(info_clips[clip_ind][frame_ind])
                try:
                    rows,cols,depth = img.shape
                    width, height = img.shape[1],img.shape[0]
                except Exception as e:
                    print('info of image that can not be read:')
                    print(info_clips[clip_ind][frame_ind])
                    print(clip_ind,frame_ind)
                    raise e

                ### mark target mouse 
                pose_toDraw = pose_clips[clip_ind][frame_ind]
                for idx_c in range(pose_toDraw.shape[0]):
                    if(pose_toDraw[idx_c,0]==0 and pose_toDraw[idx_c,1]==0):
                        continue
                    cv2.circle(img,\
                        center=(int(np.clip(pose_toDraw[idx_c,0],0,width)), int(np.clip(pose_toDraw[idx_c,1],0,height))),\
                        radius=1,\
                        color=(255,0,0),\
                        thickness=-1)
                pairs = arg.joint_pair
                for idx in range(len(pairs)):
                    if (pose_toDraw[pairs[idx][0],0]==0 and pose_toDraw[pairs[idx][0],1]==0) \
                    or (pose_toDraw[pairs[idx][1],0]==0 and pose_toDraw[pairs[idx][1],1]==0):
                        continue
                    cv2.line(img, \
                        pt1=(int(np.clip(pose_toDraw[pairs[idx][0],0],0,width)),int(np.clip(pose_toDraw[pairs[idx][0],1],0,height))), \
                        pt2=(int(np.clip(pose_toDraw[pairs[idx][1],0],0,width)),int(np.clip(pose_toDraw[pairs[idx][1],1],0,height))), \
                        color=(255,0,0), \
                        thickness=2
                        )
                pose_toDraw = poseTheOther_clips[clip_ind][frame_ind]
                for idx_c in range(pose_toDraw.shape[0]):
                    if(pose_toDraw[idx_c,0]==0 and pose_toDraw[idx_c,1]==0):
                        continue
                    cv2.circle(img,\
                        center=(int(np.clip(pose_toDraw[idx_c,0],0,width)), int(np.clip(pose_toDraw[idx_c,1],0,height))),\
                        radius=1,\
                        color=(0,0,255),\
                        thickness=-1)
                for idx in range(len(pairs)):
                    if (pose_toDraw[pairs[idx][0],0]==0 and pose_toDraw[pairs[idx][0],1]==0) \
                    or (pose_toDraw[pairs[idx][1],0]==0 and pose_toDraw[pairs[idx][1],1]==0):
                        continue
                    cv2.line(img, \
                        pt1=(int(np.clip(pose_toDraw[pairs[idx][0],0],0,width)),int(np.clip(pose_toDraw[pairs[idx][0],1],0,height))), \
                        pt2=(int(np.clip(pose_toDraw[pairs[idx][1],0],0,width)),int(np.clip(pose_toDraw[pairs[idx][1],1],0,height))), \
                        color=(0,0,255), \
                        thickness=2
                        )

                ## target mouse
                ### translation
                M = np.float32([[1,0,int(width/2-nose[0])],[0,1,int(height/2-nose[1])]])
                tra = cv2.warpAffine(img,M,(cols,rows))
                ### rotate
                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                rot = cv2.warpAffine(tra,M,(cols,rows))
                crop_target = rot[max(0,int(height/2 - ori_crop_halfHeight)):int(height/2 + ori_crop_halfHeight),\
                    max(0,int(width/2 - ori_crop_halfWidth)) : int(width/2 + ori_crop_halfWidth)\
                                 ].copy()
                crop_target = cv2.resize(crop_target, (final_crop_size, final_crop_size), interpolation=cv2.INTER_CUBIC)
                cv2.putText(crop_target,'target mouse',
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (255,0,0),
                    arg.lineType)
                ## refer mouse
                ### translation
                M = np.float32([[1,0,int(width/2-nose_ref[0])],[0,1,int(height/2-nose_ref[1])]])
                tra = cv2.warpAffine(img,M,(cols,rows))
                ### rotate
                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle_ref,1)
                rot = cv2.warpAffine(tra,M,(cols,rows))
                crop_ref = rot[max(0,int(height/2 - ori_crop_halfHeight)):int(height/2 + ori_crop_halfHeight),\
                    max(0,int(width/2 - ori_crop_halfWidth)) : int(width/2 + ori_crop_halfWidth)\
                                 ].copy()
                crop_ref = cv2.resize(crop_ref, (final_crop_size, final_crop_size), interpolation=cv2.INTER_CUBIC)
                cv2.putText(crop_ref,'refer mouse',
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (0,0,255),
                    arg.lineType)
                ## the ori frame
                img = cv2.resize(img,(ori_img_x,ori_img_y))
                cv2.putText(img,'cluster_{}_clip_{}'.format(cluster_ids_s,clip_ind),
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    arg.fontColor,
                    arg.lineType)
                

                crop = cv2.vconcat((crop_target,crop_ref))
                img = cv2.hconcat((img,crop))

                t51 = time.time()
                raw = cv2.imread('./tmp_image/raw_temp_%s.png'%(video_name_suffix))
                raw = cv2.resize(raw,(raw_x,raw_y))
                # img = cv2.hconcat((crop,raw))

                f_dis1 = cv2.imread('./tmp_image/feature_distance1_%s_%d.png'%(video_name_suffix,cluster_num_id))
                f_dis1 = cv2.resize(f_dis1,(feature_x,feature_y))
                raw_and_Feature = cv2.hconcat((raw,f_dis1))

                img = cv2.vconcat((img,raw_and_Feature))

                num_pic_v = len(cluster_ids)*2
                dend = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
                dend = cv2.resize(dend,(dendrogram_x,dendrogram_y-(num_pic_v-1)*int(dendrogram_y/num_pic_v)))
                cv2.putText(dend,cluster_ids_name[cluster_ids[0]],
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (0,0,255),
                    arg.lineType)
                tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_ids[0]))
                tsne = cv2.resize(tsne,(dendrogram_x,int(dendrogram_y/num_pic_v)))
                # dend = cv2.vconcat((dend_begin,dend))
                dend = cv2.vconcat((dend,tsne))
                for cluster_num_id in cluster_ids[1:]:
                    dend1 = cv2.imread('./tmp_image/dendrogram_%s_%d.png'%(video_name_suffix,cluster_num_id))
                    dend1 = cv2.resize(dend1,(dendrogram_x,int(dendrogram_y/num_pic_v)))
                    cv2.putText(dend1,cluster_ids_name[cluster_num_id],
                    arg.bottomLeftCornerOfText,
                    arg.font,
                    arg.fontScale,
                    (0,0,255),
                    arg.lineType)
                    dend = cv2.vconcat((dend,dend1))
                    tsne = cv2.imread('./tmp_image/tsne_%s_%d.png'%(video_name_suffix,cluster_num_id))
                    tsne = cv2.resize(tsne,(dendrogram_x,int(dendrogram_y/num_pic_v)))
                    dend = cv2.vconcat((dend,tsne))
                img = cv2.hconcat((img,dend))


                out.write(img)
                out_all.write(img)

            t6 = time.time()
            # print(10,t01-t0)
            # print(11,t1-t01)
            # print(2,t2-t1)
            # print(3,t3-t2)
            # print(4,t4-t3)
            # print(5,t5-t4)
            # print(60,t51-t5)
            # print(61,t6-t51)
        out.release()


    out_all.release()

def get_oneVideo_frameCount(video_path):
    cap = cv2.VideoCapture(video_path)
    read_flag, frame = cap.read()
    i = 0
    while(read_flag):
        read_flag, frame = cap.read()
        print('\r counting frame of %s: %d...'%(video_path,i),end='')
        i = i+1
    print('')
    return i

def init_Videos_frameLabelDict(arg,gt_dict):
    print('initialiing frame label list dict (%d videos in tatal)...'%(len(arg.videodir)))
    print('note: the key of the dict is generated with path in arg.videodir. please make sure the path can be correctly re.')
    allVideo_frameLabel_dict = {}
    for vp in arg.videodir:
        video_name = vp.split('/')[-1].split('.')[0]
        if video_name not in gt_dict:
            continue
        video_frame_count = get_oneVideo_frameCount(vp)
        allVideo_frameLabel_dict[video_name] = ['' for i in range(video_frame_count)]
    return allVideo_frameLabel_dict

def label_videos(arg, all_result, cluster_ids, info_clips, gt_dict):
    print('label with :',cluster_ids)
    allVideo_frameLabel_dict = init_Videos_frameLabelDict(arg, gt_dict)
    for clip_idx in range(len(info_clips)):
        print('\r labeling clip %d/%d'%(clip_idx,len(info_clips)),end='')
        clip_label = ''
        for c_id in cluster_ids:
            clip_label = clip_label + str(all_result[c_id]['cluster_result_fcluster'][clip_idx]) + '_'
        clip_label = clip_label[:-1]

        clip_video_name = info_clips[clip_idx][0].split('/')[-3]
        if clip_video_name not in gt_dict:
            continue
        clip_frame_ids = [int(fp.split('/')[-1].split('.')[0].split('_')[1]) for fp in info_clips[clip_idx]]
        for frame_idx in range(min(clip_frame_ids),max(clip_frame_ids)):
            allVideo_frameLabel_dict[clip_video_name][frame_idx] = clip_label
    print('')
    return allVideo_frameLabel_dict

def read_gt(file_name):
    with open(file_name,'r') as fin:
        lines = fin.readlines()
        gt = [l[:-1].split('\t') for l in lines]
    return gt

def cluster_eval(gt_dict,allVideo_frameLabel_dict,gt_mouse_idx):
    same_as_gt_count = 0
    total_predict = 0
    for video_key in gt_dict:
        all_cluster_frame_dict = {}
        for second_idx in range(len(gt_dict[video_key])):
            second_label = gt_dict[video_key][second_idx][gt_mouse_idx]
            if second_label in all_cluster_frame_dict:
                all_cluster_frame_dict[second_label] += [i for i in range(second_idx*30,(second_idx+1)*30)]
            else:
                all_cluster_frame_dict[second_label] = [i for i in range(second_idx*30,(second_idx+1)*30)]
        
        pred_cluster_dict = {}
        for cluster_label in all_cluster_frame_dict:
            all_frameId_inTheSameCluster = all_cluster_frame_dict[cluster_label]
            for frame_id in range(len(all_frameId_inTheSameCluster)):
                frame_pred_label = allVideo_frameLabel_dict[video_key][frame_id]
                if frame_pred_label == '':
                    continue
                if frame_pred_label not in pred_cluster_dict:
                    pred_cluster_dict[frame_pred_label] = 1
                    total_predict += 1
                else:
                    pred_cluster_dict[frame_pred_label] += 1
                    total_predict += 1

        print(total_predict,video_key,[[k,pred_cluster_dict[k]] for k in pred_cluster_dict]+[0])
        same_as_gt_count += max([pred_cluster_dict[k] for k in pred_cluster_dict]+[0])

    print('total prediction: %d \nprediction same as gt:%d'%(total_predict,same_as_gt_count))

def get_normFeature_byArg(feature_arg_dict,feature_clips_dict):
    if feature_arg_dict['feat_key'] not in feature_clips_dict:
        print('Feature {} does not exists!'.format(feature_arg_dict['feat_key']))
        print('All feature:')
        print(list(feature_clips_dict.keys()))
        raise 
    feat = feature_clips_dict[feature_arg_dict['feat_key']]
    feat_norm = name_norm(feat,feature_arg_dict['norm'])
    return feat_norm

def get_result_for_evaluation(arg,all_result_dict,all_info_selected,feature_clips_dict,video_id_forEvaluation,c_arg_dict_idx_forFeature):
    if isinstance(video_id_forEvaluation,int): video_id_forEvaluation=str(video_id_forEvaluation)
    c_arg_dict = arg.cluster_arg[c_arg_dict_idx_forFeature]
    ### Normalize and concatenate feature  
    feat_norm = c_arg_dict['features_arg'][0]['weight']*get_normFeature_byArg(c_arg_dict['features_arg'][0],feature_clips_dict)
    for f_i in range(1,len(c_arg_dict['features_arg'])):
        feat_norm_tmp = c_arg_dict['features_arg'][f_i]['weight']*get_normFeature_byArg(c_arg_dict['features_arg'][f_i],feature_clips_dict)
        feat_norm = np.concatenate([feat_norm,feat_norm_tmp] ,axis = 1)

    clip_indexes = []
    features_norm = []
    predictions = []
    info_clips = []
    for i in range(len(all_info_selected['info_clips'])):
        if all_info_selected['info_clips'][i]['clip_id'].split('_')[0] == video_id_forEvaluation:
            clip_indexes.append(i)
            features_norm.append(feat_norm[i,:])
            predictions.append(all_info_selected['info_clips'][i]['cluster_id'])
            info_clips.append(all_info_selected['info_clips'][i])
    return {'clip_indexes':clip_indexes,
        'features_norm':features_norm,
        'predictions':predictions,
        'info_clips':info_clips}


def strList_to_intList(strList,label_per_secound=1):
    uni_strs = np.unique(strList)
    strs_dict = {}
    for i in range(len(uni_strs)):
        strs_dict[uni_strs[i]] = i+1
    intList = []
    for i in range(len(strList)):
        for j in range(label_per_secound):
            intList.append(strs_dict[strList[i]])

    return intList

def expand_list(l,expand_ratio):
    l_expanded = []
    for i in range(len(l)):
        for j in range(expand_ratio):
            l_expanded.append(l[i])
    return l_expanded

def evaluate(arg,all_result_dict,all_info_selected,feature_clips_dict,gt_key = ['gt1_1929_black_two'],gt_per_secound=2, exclude_unknown = True,gt_file = '/disk2/zexin/data/mice/behavior_data/Alphapose Ethogram Scoring Sheet.csv'):
    assert(len(gt_key)==len(arg.video_ids_forEvaluation))
    ##################### load gt #####################
    gt1_1411_black_two = []
    gt2_1411_black_two = []
    gt1_1929_black_two = []
    gt2_1929_black_two = []
    csv_list = []

    with open(gt_file,'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            csv_list.append(row)

    for l in csv_list[6:]:
        if l[-1] != '':
            gt1_1929_black_two.append(l[-2])
            gt2_1929_black_two.append(l[-1])
        if l[-3] != '':
            gt1_1411_black_two.append(l[-4])
            gt2_1411_black_two.append(l[-3])

    gts={}
    # gts['gt1_1411_black_two'] = strList_to_intList(gt1_1411_black_two,gt_per_secound=2)
    # gts['gt2_1411_black_two'] = strList_to_intList(gt2_1411_black_two,gt_per_secound=2)
    # gts['gt1_1929_black_two'] = strList_to_intList(gt1_1929_black_two,gt_per_secound=2)
    # gts['gt2_1929_black_two'] = strList_to_intList(gt2_1929_black_two,gt_per_secound=2)
    gts['gt1_1411_black_two'] = gt1_1411_black_two
    gts['gt2_1411_black_two'] = gt2_1411_black_two
    gts['gt1_1929_black_two'] = gt1_1929_black_two
    gts['gt2_1929_black_two'] = gt2_1929_black_two

    gt_all = []
    pred_all = []
    for e_i in range(len(gt_key)):
        print('key:',gt_key)
        print('length of gt loaded:',len(gts[gt_key[e_i]]))
        gt = expand_list(gts[gt_key[e_i]],expand_ratio=gt_per_secound)
        ##################### load pred #####################
        ### filter videos without gt
        evaluate_dict = get_result_for_evaluation(arg,\
            all_result_dict,all_info_selected,feature_clips_dict,\
            video_id_forEvaluation=arg.video_ids_forEvaluation[e_i],\
            c_arg_dict_idx_forFeature=0)

        ### process prediction to make it match gt
        pred_forEvaluation = ['unknown' for i in range(len(gt))]
        pred_count = 0
        for i in range(len(evaluate_dict['info_clips'])):
            clip_index = int(evaluate_dict['info_clips'][i]['clip_id'].split('_')[-1])
            cluster_str = str(evaluate_dict['info_clips'][i]['cluster_id'])
            pred_forEvaluation[clip_index] = cluster_str
            pred_count += 1

        # print(pred_forEvaluation)
        if exclude_unknown:
            gt_tmp = []
            pred = []
            for i in range(len(pred_forEvaluation)):
                if pred_forEvaluation[i]!='unknown':
                    pred.append(pred_forEvaluation[i])
                    gt_tmp.append(gt[i])
            gt = gt_tmp
        else:
            pred = pred_forEvaluation

        gt_all += gt
        pred_all += pred

    print('final length of gt to evaluate:',len(gt_all))
    gt = strList_to_intList(gt_all,label_per_secound=1)
    pred = strList_to_intList(pred_all,label_per_secound=1)

    ####### the enclosed part is for experiment only #########
    pred_random = [random.randint(1,25) for i in range(len(gt))]
    print('Adjusted Rand index of random assignments:%f'%metrics.adjusted_rand_score(gt, pred_random))

    ####### the enclosed part is for experiment only #########

    print('###############evaluation results######################')
    ## Bounded range [-1, 1]: negative values are bad (independent labelings), similar clusterings have a positive ARI, 1.0 is the perfect match score.
    print('Adjusted Rand index:%f'%metrics.adjusted_rand_score(gt, pred))

    ## Upper bound of 1: Values close to zero indicate two label assignments that are largely independent, while values close to one indicate significant agreement. Further, an AMI of exactly 1 indicates that the two label assignments are equal (with or without permutation).
    print('Mutual Information based scores:%f'%metrics.adjusted_mutual_info_score(gt, pred))

    ## 0.0 is as bad as it can be, 1.0 is a perfect score.
    print('Homogeneity, completeness and V-measure:{}'.format(metrics.homogeneity_completeness_v_measure(gt, pred)))

    ## Values close to zero indicate two label assignments that are largely independent, while values close to one indicate significant agreement. Further, values of exactly 0 indicate purely independent label assignments and a FMI of exactly 1 indicates that the two label assignments are equal (with or without permutation).
    print('Fowlkes-Mallows scores:{}'.format(metrics.fowlkes_mallows_score(gt, pred)))

    print('#######################################################')
    return {'Adjusted Rand index':metrics.adjusted_rand_score(gt, pred),\
    'Mutual Information based scores':metrics.adjusted_mutual_info_score(gt, pred),\
    'Homogeneity':metrics.homogeneity_completeness_v_measure(gt, pred)[0],\
    'completeness':metrics.homogeneity_completeness_v_measure(gt, pred)[1],\
    'V-measure':metrics.homogeneity_completeness_v_measure(gt, pred)[2],\
    'Fowlkes-Mallows scores':metrics.fowlkes_mallows_score(gt, pred),\
    'gt':gt,\
    'pred':pred
    }

def load_gt(gt_key = 'gt1_1929_black_two',exclude_unknown = True,gt_file = '/disk2/zexin/data/mice/behavior_data/Alphapose Ethogram Scoring Sheet.csv'):
    ##################### load gt #####################
    gt1_1411_black_two = []
    gt2_1411_black_two = []
    gt1_1929_black_two = []
    gt2_1929_black_two = []
    csv_list = []

    with open(gt_file,'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            csv_list.append(row)

    for l in csv_list[6:]:
        if l[-1] != '':
            gt1_1929_black_two.append(l[-2])
            gt2_1929_black_two.append(l[-1])
        if l[-3] != '':
            gt1_1411_black_two.append(l[-4])
            gt2_1411_black_two.append(l[-3])

    gts={}
    gts['gt1_1411_black_two'] = strList_to_intList(gt1_1411_black_two,gt_per_secound=2)
    gts['gt2_1411_black_two'] = strList_to_intList(gt2_1411_black_two,gt_per_secound=2)
    gts['gt1_1929_black_two'] = strList_to_intList(gt1_1929_black_two,gt_per_secound=2)
    gts['gt2_1929_black_two'] = strList_to_intList(gt2_1929_black_two,gt_per_secound=2)
    
    return gts

def evaluate_inFunction(arg,all_result_dict,all_info_selected,feature_clips_dict,gts,gt_key = 'gt1_1929_black_two',exclude_unknown = True):
    print('length of gt:',len(gts[gt_key]))

    ##################### load pred #####################
    ### filter videos without gt
    evaluate_dict = get_result_for_evaluation(arg,all_result_dict,all_info_selected,feature_clips_dict,c_arg_dict_idx_forFeature=0)

    ### process prediction to make it match gt
    pred_forEvaluation = ['unknown' for i in range(len(gts[gt_key]))]
    pred_count = 0
    for i in range(len(evaluate_dict['info_clips'])):
        clip_index = int(evaluate_dict['info_clips'][i]['clip_id'].split('_')[-1])
        cluster_str = str(evaluate_dict['info_clips'][i]['cluster_id'])
        pred_forEvaluation[clip_index] = cluster_str
        pred_count += 1

    print('length of pred:',pred_count)

    # print(pred_forEvaluation)
    if exclude_unknown:
        gt = []
        pred = []
        for i in range(len(pred_forEvaluation)):
            if pred_forEvaluation[i]!='unknown':
                pred.append(pred_forEvaluation[i])
                gt.append(gts[gt_key][i])
    else:
        gt = gts[gt_key]
        pred = pred_forEvaluation
    print('final length:',len(gt))
    pred_forEvaluation = strList_to_intList(pred_forEvaluation,gt_per_secound=1)


    print('evaluating %s'%gt_key)

    ## Bounded range [-1, 1]: negative values are bad (independent labelings), similar clusterings have a positive ARI, 1.0 is the perfect match score.
    print('Adjusted Rand index:%f'%metrics.adjusted_rand_score(gt, pred))

    ## Upper bound of 1: Values close to zero indicate two label assignments that are largely independent, while values close to one indicate significant agreement. Further, an AMI of exactly 1 indicates that the two label assignments are equal (with or without permutation).
    print('Mutual Information based scores:%f'%metrics.adjusted_mutual_info_score(gt, pred))

    ## 0.0 is as bad as it can be, 1.0 is a perfect score.
    print('Homogeneity, completeness and V-measure:{}'.format(metrics.homogeneity_completeness_v_measure(gt, pred)))

    ## Values close to zero indicate two label assignments that are largely independent, while values close to one indicate significant agreement. Further, values of exactly 0 indicate purely independent label assignments and a FMI of exactly 1 indicates that the two label assignments are equal (with or without permutation).
    print('Fowlkes-Mallows scores:{}'.format(metrics.fowlkes_mallows_score(gt, pred)))

    return {'Adjusted Rand index':metrics.adjusted_rand_score(gt, pred),\
    'Mutual Information based scores':metrics.adjusted_mutual_info_score(gt, pred),\
    'Homogeneity':metrics.homogeneity_completeness_v_measure(gt, pred)[0],\
    'completeness':metrics.homogeneity_completeness_v_measure(gt, pred)[1],\
    'V-measure':metrics.homogeneity_completeness_v_measure(gt, pred)[2],\
    'Fowlkes-Mallows scores':metrics.fowlkes_mallows_score(gt, pred),\
    }

