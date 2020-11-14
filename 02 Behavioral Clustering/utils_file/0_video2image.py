import numpy as np
import cv2
import json
import math
import copy
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from tqdm import tqdm
from contour_utils import mkdir_p

import setting

def get_frames(video_path,pose_track_vis_path):
    cap = cv2.VideoCapture(video_path)
    read_flag, frame = cap.read()
    if not read_flag:
        print('read '+video_path+' failed!')
    width, height,depth = np.asarray(frame).shape

    i = 0
    if not os.path.exists(pose_track_vis_path):
        mkdir_p(pose_track_vis_path)

    while(read_flag):
        cv2.imwrite(pose_track_vis_path+ '/frame_{}.png'.format(i),frame)
        print('\r {}/frame_{}.png'.format(pose_track_vis_path,i),end='')


        read_flag, frame = cap.read()
        i = i+1



if __name__ == '__main__':
    arg = setting.args_class()

    for video_path, pose_track_vis_path in zip(arg.videodir,arg.imgdir):
        print('generating %s'%(video_path))
        get_frames(video_path,pose_track_vis_path)
    



