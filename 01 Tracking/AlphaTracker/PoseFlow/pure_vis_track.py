import numpy as np
import os
import json
import copy
import heapq
from munkres import Munkres, print_matrix
from PIL import Image
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import *
from matching import orb_matching
import argparse
import cv2
from functools import cmp_to_key
import time
import queue as Queue


def display_pose_cv2(imgdir, visdir, tracked, cmap, args):

    print("Start visualization...\n")
    colors =['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
    part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    pairs = [[0,1],[0,2],[0,3]]
    # min_frameNo = np.min([int(k) for k in tracked.keys()])
    
    img_tmp = cv2.imread(os.path.join(imgdir,args.image_format%(list(tracked.keys())[0])))
    height, width, channels = img_tmp.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    # out = cv2.VideoWriter(args.out_video_path, fourcc, 20.0, (width, height))
    out = cv2.VideoWriter(args.out_video_path, fourcc, 20.0, (width, height))
   
    aa = list(tracked.keys())
    # aa.sort(cmp=lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1]))
    aa.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))
    # aa.sort(cmp=lambda a,b:int(a)-int(b))
    # aa.sort()
    qs = [ Queue.Queue(maxsize=30) for i in range(args.max_pid_id_setting+1)]
    for imgname in tqdm(aa):
        img = cv2.imread(os.path.join(imgdir,args.image_format%(imgname)))
        width, height = img.shape[1],img.shape[0]
        cv2.putText(img, \
                        text=imgname, \
                        org=(50,50), \
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                        fontScale=1, \
                        color=(255,255,255), \
                        thickness=3)
        for pid in range(len(tracked[imgname])):
            pose = np.array(tracked[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
            tracked_id = tracked[imgname][pid]['idx']
#             cv2.putText(img, \
#                         text=str(tracked_id), \
#                         org=(int(np.clip(pose[0,0],0,width)), int(np.clip(pose[0,1],0,height))), \
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
#                         fontScale=int(1*np.mean(pose[:,2])+1), \
#                         color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), \
#                         thickness=3)
            
#             for idx_c in range(pose.shape[0]):
#                 cv2.circle(img,\
#                     center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),\
# #                     radius=int(20*np.mean(pose[:,2])+1),\
#                     radius=3,\
#                     color=(255,(80*idx_c)%255,0),\
#                     thickness=-1)
#             for idx in range(len(pairs)):
#                 cv2.line(img, \
#                     pt1=(int(np.clip(pose[pairs[idx][0],0],0,width)),int(np.clip(pose[pairs[idx][0],1],0,height))), \
#                     pt2=(int(np.clip(pose[pairs[idx][1],0],0,width)),int(np.clip(pose[pairs[idx][1],1],0,height))), \
# #                     color=cmap(tracked_id*5), \
#                     color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), \
# #                     thickness=int(20*np.mean(pose[pairs[idx],2])+1)
#                     thickness=3
#                     )
            if qs[tracked_id].full():
                qs[tracked_id].get()
            qs[tracked_id].put(\
                (
                    (int(np.clip((tracked[imgname][pid]['box'][0]+tracked[imgname][pid]['box'][2])/2,0,width))), \
                    (int(np.clip((tracked[imgname][pid]['box'][1]+tracked[imgname][pid]['box'][3])/2,0,width)))\
                 # int(np.clip(pose[0,1],0,height)))
                )
                )

        ## draw track
        for q_id in range(len(qs)):
            q = qs[q_id]
            q_tmp = Queue.Queue(maxsize=30)
            if q.empty():
                continue
            s = q.get()
            q_tmp.put(s)
            while not q.empty():
                e = q.get()
                q_tmp.put(e)
                cv2.line(img, \
                    pt1=s, \
                    pt2=e, \
                    color=((160*q_id)%255,(80*q_id)%255,(30*q_id)%255), \
                    thickness=3
                    )
                s = e
            qs[q_id] = q_tmp

        if not os.path.exists(visdir): 
            os.mkdir(visdir)
        # cv2.imwrite(os.path.join(visdir,str(int(imgname.split()[0])-min_frameNo)+".png"),img)
        # cv2.imwrite(os.path.join(visdir,imgname+'.png'),img)
        out.write(img) # Write out frame to video
    print('demo image is generated in ',visdir)
    print('demo video is generated as: ',args.out_video_path)
    out.release()


parser = argparse.ArgumentParser(description='FoseFlow Tracker')
parser.add_argument('--imgdir', type=str, required=True, help="Must input the images dir")
parser.add_argument('--in_json', type=str, help="result json predicted by AlphaPose")
parser.add_argument('--out_json', type=str, required=True, help="output path of tracked json")
parser.add_argument('--visdir', type=str, default="", help="visulization tracked results of video sequences")
parser.add_argument('--image_format', type=str, default="%d.png", help="image foramt")
parser.add_argument('--out_video_path', type=str, default="output.mp4", help="image foramt")

parser.add_argument('--link', type=int, default=100)
parser.add_argument('--drop', type=float, default=2.0)
parser.add_argument('--num', type=int, default=7)
parser.add_argument('--mag', type=int, default=30)
parser.add_argument('--match', type=float, default=0.2)
parser.add_argument('--max_pid_id_setting', type=int, default=-1)
parser.add_argument('--weights', nargs='+', type=int, default=[1,2,1,2,0,0], help="dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score")


args = parser.parse_args()


tracked_json = args.out_json
image_dir = args.imgdir
vis_dir = args.visdir

with open(tracked_json,'r') as json_file:
    # json_file.write(json.dumps(notrack))
    # notrack = json.load(json_file)
    notrack = json.loads(json_file.read())
    cmap = plt.cm.get_cmap("hsv", args.max_pid_id_setting+5)
    display_pose_cv2(image_dir, vis_dir, notrack, cmap, args)


