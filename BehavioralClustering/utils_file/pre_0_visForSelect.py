# -*- coding: UTF-8 -*- 
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




arg = setting.args_class()

    # for video_path,json_path,contour_path in zip(arg.videodir,arg.tracked_json,arg.contdir):
for i in range(len(arg.videodir)):
# for i in [1,2,3,4,5]:
# for i in [1]:
    video_path = arg.videodir[i]
    json_path = arg.tracked_json[i]
    contour_path = arg.contdir[i]
    #     get_samples(video_path,json_path,contour_path)

    # ------------------- video for debug -----------------------
    out_video_path = '{}/contour_{}_debug.mp4'.format('./tmp',video_path.split('/')[-1].split('.')[0])
    print('\n'+out_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_all = cv2.VideoWriter(out_video_path, fourcc, 9.0, (800+300,300+300))
    out_all = cv2.VideoWriter(out_video_path, fourcc, 9.0, (1100,600))



    data = contour_utils.load_json(json_path)
    # ------------------- Read First Frame -----------------------

    # cap = cv2.VideoCapture(dir_name, video_name)
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    read_flag, frame = cap.read()
    width, height,depth = np.asarray(frame).shape

    i = 0
    if not os.path.exists(contour_path):
        mkdir_p(contour_path)

    db = False
    show_count = 0
    plt.figure(figsize = (15,5))


    # ----------------- Sample all the mouse mask ----------------------------
    while(read_flag):
        # if (i<18049) : ## bad frame
        #     read_flag, frame = cap.read()
        #     i += 1
        #     continue
        frame_ori = frame
        # print(frame.shape)


        # 当前frame的pose信息
        bad_clip = False
        mouses = data['frame_{}'.format(i)]
        for m in mouses:
            if not ('keypoints' in m):
                bad_clip = True
                break
        if bad_clip:
            print('\n frame '+str(i)+' of '+out_video_path+' does not have enough mice')
            read_flag, frame = cap.read()
            i += 1
            continue
        # if (len(mouses)<2) : ## bad frame
        #     read_flag, frame = cap.read()
        #     i += 1
        #     continue
        for m_id in range(2):
            p = np.asarray(mouses[m_id]['keypoints']).reshape((4,3))
            for p_id in [0,3]:
                if p[p_id,0]<0 or p[p_id,1]>width or p[p_id,1]<0 or p[p_id,0]>height: ## bad frame
                    bad_clip = True
                    break
        if bad_clip:
            print('\n frame '+str(i)+' of '+out_video_path+' has out picture point')
            read_flag, frame = cap.read()
            i += 1
            continue

        pose1 = np.asarray(mouses[0]['keypoints']).reshape((4,3))
        pose2 = np.asarray(mouses[1]['keypoints']).reshape((4,3))


        # 当前frame中寻找contours
        frame = gaussian_filter(frame, sigma=3)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        ret,thre = cv2.threshold(gray,40,255,0)
        contours, hierarchy = cv2.findContours(thre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]   ## different cv version
        # binary, contours, hierarchy = cv2.findContours(thre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        # 遍历每个contour，看是否符合要求
        for contour_id, contour in enumerate(contours):

            if (contour.size>150) and (contour.size<600):

                # 把contour以binary mask的形式呈现
                mask = np.zeros((width,height,depth),dtype = 'uint8')
                cv2.drawContours(mask, contours, contour_id, (255,255,255), -1)

                # 假设当前的contour符合要求，但发现有任意一个keypoint不在mask内，就放弃
                flag = True
                for j in [0,3]:
                    if (mask[int(pose1[j,1]),int(pose1[j,0]),0] == 0):
                        flag = False
                        continue

                if flag:

                    # 假设当前的contour符合要求，但发现有另一只老鼠的keypoint在mask内，就放弃
                    flag2 = True
                    for j in [0,3]:
                        if (mask[int(pose2[j,1]),int(pose2[j,0]),0] > 0):
                            flag2 = False
                            continue

                    if flag2:

                        # 首先把mask平移到中心
                        rows,cols,depth = mask.shape
                        x,y,w,h = cv2.boundingRect(contour)
                        M = np.float32([[1,0,640-(x+w/2)],[0,1,360-(y+h/2)]])
                        tra = cv2.warpAffine(mask,M,(cols,rows))
                        tra_f = cv2.warpAffine(frame_ori,M,(cols,rows))

                        # 旋转到身体的轴在x轴上
                        body = pose1[3,0:2]-pose1[0,0:2]
                        rho,phi = contour_utils.cart2pol(body[0],body[1])
                        angle = math.degrees(phi)

                        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                        rot = cv2.warpAffine(tra,M,(cols,rows))
                        rot_f = cv2.warpAffine(tra_f,M,(cols,rows))

                        # 裁剪成 200 * 200
                        crop = rot[260:460,540:740].copy()
                        crop_f = rot_f[260:460,540:740].copy()

                        # # show image 
                        # plt.imshow(crop)
                        # plt.show()
                        # # plt.close()
                        # # plt.figure(figsize = (15,5))
                        # plt.imshow(crop_f)
                        # plt.show()
                        # # plt.close()
                        crop = cv2.resize(crop, (300,300), interpolation=cv2.INTER_AREA)  
                        crop_f = cv2.resize(crop_f, (300,300), interpolation=cv2.INTER_AREA)  
                        comb_img = cv2.vconcat((crop,crop_f))
                        # plt.imshow(comb_img)
                        # plt.show()

                        # 可视化原图和pose
                        cv2.putText(frame_ori, \
                            text=video_path.split('/')[-1].split('.')[0]+'_'+str(i), \
                            org=(50,50), \
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                            fontScale=1, \
                            color=(255,255,255), \
                            thickness=3)
                        pairs = [[0,ii] for ii in range(1,pose1.shape[0])]
                        tracked_id = 0
                        for pose in [pose1,pose2]:
                            tracked_id += 1
                            for idx in range(len(pairs)):
                                cv2.line(frame_ori, \
                                    pt1=(int(np.clip(pose[pairs[idx][0],0],0,height)),int(np.clip(pose[pairs[idx][0],1],0,width))), \
                                    pt2=(int(np.clip(pose[pairs[idx][1],0],0,height)),int(np.clip(pose[pairs[idx][1],1],0,width))), \
                                    # color=cmap(tracked_id*5), \
                                    color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), \
                                    # thickness=int(20*np.mean(pose[pairs[idx],2])+1)
                                    thickness=3
                                    )
                                cv2.putText(frame_ori, \
                                    text=str(tracked_id), \
                                    org=(int(np.clip(pose[0,0],0,height)), int(np.clip(pose[0,1],0,width))), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                                    fontScale=int(1*np.mean(pose[:,2])+1), \
                                    color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), \
                                    thickness=3)
                        frame_ori = cv2.resize(frame_ori, (800,600), interpolation=cv2.INTER_AREA)  
#                         plt.imshow(frame_ori)
#                         plt.show()
                        
#                         print(frame_ori.shape,comb_img.shape)
                        comb_img = cv2.hconcat((frame_ori,comb_img))
                        # print(frame_ori.shape,comb_img.shape)

                        out_all.write(comb_img)
#                         plt.imshow(comb_img)
#                         plt.show()

                        
                       
    #                     cv2.imwrite(contour_path+ '/mask_{}.png'.format(i),crop)
                        print('\r {} {}/mask_{}.png'.format(show_count, contour_path,i),end='',flush=True)
                        
                        show_count+=1
                        # if(show_count>9*30*1):
                            # db = True

                        continue

        read_flag, frame = cap.read()
        i += 1
        # print(i)
        if db:
            break
    out_all.release()

    cap.release()
    plt.close()









