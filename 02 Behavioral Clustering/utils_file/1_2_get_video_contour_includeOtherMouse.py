import numpy as np
import cv2
import json
import math
import copy
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from tqdm import tqdm
from contour_utils import mkdir_p

import contour_utils

import setting



def get_samples(video_path,json_path,contour_path,arg,targetMouseID):
    # data = contour_utils.load_json(json_name)
    data = contour_utils.load_json(json_path)
    # contour_path = dir_name + 'contour_zexin'

    # ------------------- Read First Frame -----------------------

    # cap = cv2.VideoCapture(dir_name, video_name)
    print('getting contour of %s'%video_path)
    cap = cv2.VideoCapture(video_path)
    read_flag, frame = cap.read()
    width, height,depth = np.asarray(frame).shape

    i = 0
    if not os.path.exists(contour_path):
        mkdir_p(contour_path)


    # ----------------- Sample all the mouse mask ----------------------------
    count_lessMouse = 0
    count_out = 0
    count_used = 0
    count_frame = 0
    a,b = 0,0
    while(read_flag):
        count_frame += 1
        # if (i<625):
        #     read_flag, frame = cap.read()
        #     i += 1
        #     continue
        bad_clip = False
        try:
            mouses = data['frame_{}'.format(i)]
        except:
            read_flag, frame = cap.read()
            i += 1
            continue
        for m in mouses:
            if not ('keypoints' in m):
                bad_clip = True
                break
        if bad_clip:
            count_lessMouse += 1
            print('\r frame '+str(i)+' of '+video_path+' does not have enough mice! (%d less, %d out, %d used,%d frames)'%(count_lessMouse,count_out,count_used,count_frame),end='')
            read_flag, frame = cap.read()
            i += 1
            continue
        for m_id in range(2):
            p = np.asarray(mouses[m_id]['keypoints']).reshape((arg.joint_num,3))
            for p_id in [0,3]:
                if p[p_id,0]<0 or p[p_id,1]>width or p[p_id,1]<0 or p[p_id,0]>height: ## bad frame
                    bad_clip = True
                    break
        if bad_clip:
            count_out += 1
            print('\r frame '+str(i)+' of '+video_path+' has out picture point!(%d less, %d out, %d used,%d frames)'%(count_lessMouse,count_out,count_used,count_frame),end='')
            read_flag, frame = cap.read()
            i += 1
            continue

        # 当前frame的pose信息
        mouses = data['frame_{}'.format(i)]
        if targetMouseID == 0:
            pose1 = np.asarray(mouses[0]['keypoints']).reshape((arg.joint_num,3))
            pose2 = np.asarray(mouses[1]['keypoints']).reshape((arg.joint_num,3))
        else:
            pose1 = np.asarray(mouses[1]['keypoints']).reshape((arg.joint_num,3))
            pose2 = np.asarray(mouses[0]['keypoints']).reshape((arg.joint_num,3))
        

        # 当前frame中寻找contours
        frame = gaussian_filter(frame, sigma=3)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        ret,thre = cv2.threshold(gray,50,255,0)
        
        contours, hierarchy = cv2.findContours(thre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # 遍历每个contour，看是否符合要求
        for contour_id, contour in enumerate(contours):

    #         if (contour.size>150) and (contour.size<600):
            if (contour.size>150):

                # 把contour以binary mask的形式呈现
                mask = np.zeros((width,height,depth),dtype = 'uint8')
                cv2.drawContours(mask, contours, contour_id, (255,255,255), -1)
                
                if(np.sum(mask==255)>width*height/2):
                    # mask too large, may be the background
                    continue
                
                # 假设当前的contour符合要求，但发现有任意一个keypoint不在mask内，就放弃
                flag = True
                for j in [0,3]:
                    if (mask[int(pose1[j,1]),int(pose1[j,0]),0] == 0):
                        continue

                if flag:

                    # 首先把mask平移到中心
                    rows,cols,depth = mask.shape
                    x,y,w,h = cv2.boundingRect(contour)
    #                 M = np.float32([[1,0,w/2-(x+w/2)],[0,1,h/2-(y+h/2)]])
                    mouse_center_y = int((pose1[0,0]+pose1[3,0])/2)
                    mouse_center_x = int((pose1[0,1]+pose1[3,1])/2)
                    M = np.float32([[1,0,height/2-mouse_center_y],[0,1,width/2-mouse_center_x]])
                    tra = cv2.warpAffine(mask,M,(cols,rows))

                    # 旋转到身体的轴在x轴上
                    body = pose1[3,0:2]-pose1[0,0:2]
                    rho,phi = contour_utils.cart2pol(body[0],body[1])
                    angle = math.degrees(phi)
                    
                    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                    rot = cv2.warpAffine(tra,M,(cols,rows))
                    

                    # 裁剪成 200 * 200
                    ori_crop_halfSize = 400/2
                    final_crop_size = 200
                    crop = rot[int(width/2-ori_crop_halfSize):int(width/2+ori_crop_halfSize),int(height/2-ori_crop_halfSize):int(height/2+ori_crop_halfSize)].copy()
                    crop = cv2.resize(crop, (final_crop_size, final_crop_size), interpolation=cv2.INTER_CUBIC)  
                    
                    cv2.imwrite(contour_path+ '/mask_mouse{}_{}.png'.format(targetMouseID,i),crop)
                    print('\r' + contour_path+ '/mask_mouse{}_{}.png'.format(targetMouseID,i),end = '',flush=True)
                    count_used += 1
                    continue

        read_flag, frame = cap.read()
        i += 1

    cap.release()
    print('%d frames dont have 2 mice, %d frames have points outside pic. %d frame will be used, %d frame in total'%(count_lessMouse,count_out,count_used,count_frame))



if __name__ == '__main__':
    # dir_name = '/disk1/zexin/project/mice/clustering_sequencial/forZexin/results/0603/1411_black_two/'
    # video_name = '1411_black_two.mov'
    # json_name= '/disk1/zexin/project/mice/clustering_sequencial/forZexin/results/0603/1411_black_two/alphapose-results-forvis-tracked.json'
    arg = setting.args_class()

    if (len(sys.argv)!=1):
        all_video_paths_id = [int(sys.argv[1])]
        print('only deal with the %d th video path: \n '%(int(sys.argv[1])))
    else:
        all_video_paths_id = [i for i in range(len(arg.videodir))]

    for i in all_video_paths_id:
        video_path = arg.videodir[i]
        json_path = arg.tracked_json[i]
        contour_path = arg.contdir[i]
        get_samples(video_path,json_path,contour_path,arg,targetMouseID=0)
        get_samples(video_path,json_path,contour_path,arg,targetMouseID=1)




