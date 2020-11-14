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


# def get_samples(dir_name,video_name,json_name):
def get_samples(video_path,json_path,contour_path):

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
    while(read_flag):
        # if (i<625):
        #     read_flag, frame = cap.read()
        #     i += 1
        #     continue
        bad_clip = False
        mouses = data['frame_{}'.format(i)]
        for m in mouses:
            if not ('keypoints' in m):
                bad_clip = True
                break
        if bad_clip:
            print('\n frame '+str(i)+' of '+video_path+' does not have enough mice!')
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
            print('\n frame '+str(i)+' of '+video_path+' has out picture point!')
            read_flag, frame = cap.read()
            i += 1
            continue

        # 当前frame的pose信息
        mouses = data['frame_{}'.format(i)]
        # print(mouses)
        # print(len(mouses))
        # print(i)
        pose1 = np.asarray(mouses[0]['keypoints']).reshape((4,3))
        pose2 = np.asarray(mouses[1]['keypoints']).reshape((4,3))

        # 当前frame中寻找contours
        frame = gaussian_filter(frame, sigma=3)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        ret,thre = cv2.threshold(gray,40,255,0)
        contours, hierarchy = cv2.findContours(thre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        mouse_processed = []
        # 遍历每个contour，看是否符合要求
        for contour_id, contour in enumerate(contours):

            if (contour.size>150) and (contour.size<600):

                # 把contour以binary mask的形式呈现
                mask = np.zeros((width,height,depth),dtype = 'uint8')
                cv2.drawContours(mask, contours, contour_id, (255,255,255), -1)

                # 假设当前的contour符合要求，但发现有任意一个keypoint不在mask内，就放弃
                flag = True

                # 如果pose1的头和尾都在mask里面则mask算第一只老鼠的，pose2类似，否则放弃
                if (mask[int(pose1[0,1]),int(pose1[0 ,0]),0] != 0) and (mask[int(pose1[3,1]),int(pose1[3,0]),0] != 0) and (0 not in mouse_processed):
                    mouse_id = 0
                    mouse_processed.append(0)
                elif (mask[int(pose2[0,1]),int(pose2[0 ,0]),0] != 0) and (mask[int(pose2[3,1]),int(pose2[3,0]),0] != 0) and (1 not in mouse_processed):
                    mouse_id = 1
                    mouse_processed.append(1)
                elif (mask[int(pose1[0,1]),int(pose1[0 ,0]),0] != 0) or (mask[int(pose1[3,1]),int(pose1[3,0]),0] != 0) and (0 not in mouse_processed):
                    mouse_id = 0
                    mouse_processed.append(0)
                elif (mask[int(pose2[0,1]),int(pose2[0 ,0]),0] != 0) or (mask[int(pose2[3,1]),int(pose2[3,0]),0] != 0) and (1 not in mouse_processed):
                    mouse_id = 1
                    mouse_processed.append(1)
                else:
                    continue

                # 首先把mask平移到中心
                rows,cols,depth = mask.shape
                x,y,w,h = cv2.boundingRect(contour)
                M = np.float32([[1,0,640-(x+w/2)],[0,1,360-(y+h/2)]])
                tra = cv2.warpAffine(mask,M,(cols,rows))

                # 旋转到身体的轴在x轴上
                body = pose1[3,0:2]-pose1[0,0:2]
                rho,phi = contour_utils.cart2pol(body[0],body[1])
                angle = math.degrees(phi)

                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                rot = cv2.warpAffine(tra,M,(cols,rows))

                # 裁剪成 200 * 200
                crop = rot[260:460,540:740].copy()

                cv2.imwrite(contour_path+ '/mask_{}_mouse_{}.png'.format(i,mouse_id),crop)
                print('\r'+ contour_path+ '/mask_{}_mouse_{}.png'.format(i,mouse_id),end = '',flush=True)


        read_flag, frame = cap.read()
        i += 1

    cap.release()

if __name__ == '__main__':

    
    # dir_name = '/disk1/zexin/project/mice/clustering_sequencial/forZexin/results/0603/1411_black_two/'
    # video_name = '1411_black_two.mov'
    # json_name= '/disk1/zexin/project/mice/clustering_sequencial/forZexin/results/0603/1411_black_two/alphapose-results-forvis-tracked.json'
    arg = setting.args_class()

    # for video_path,json_path,contour_path in zip(arg.videodir,arg.tracked_json,arg.contdir):
    for i in range(len(arg.videodir)):
    # for i in [int(sys.argv[1])]:
        video_path = arg.videodir[i]
        json_path = arg.tracked_json[i]
        contour_path = arg.contdir[i]
        get_samples(video_path,json_path,contour_path)




