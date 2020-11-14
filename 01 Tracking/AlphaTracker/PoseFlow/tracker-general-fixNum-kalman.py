# coding: utf-8

'''
File: tracker-general.py
Project: AlphaPose
File Created: Tuesday, 18st Dec 2018 14:55:41 pm
-----
Last Modified: Thursday, 20st Dec 2018 23:24:47 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''
# %matplotlib inline

import numpy as np
import os
import json
import copy
import heapq
from munkres import Munkres, print_matrix
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from matching import orb_matching
import argparse
import cv2

from kalman_filter_multi_object_tracking.tracker import Tracker


def display_pose_cv2(imgdir, visdir, tracked, cmap, args):

    print("Start visualization...\n")
    colors =['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
    part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    pairs = [[0,1],[0,2],[0,3]]
    # min_frameNo = np.min([int(k) for k in tracked.keys()])
    
    img_tmp = cv2.imread(os.path.join(imgdir,args.image_format%(tracked.keys()[0])))
    height, width, channels = img_tmp.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(args.out_video_path, fourcc, 20.0, (width, height))
   
    aa = tracked.keys()
    aa.sort(cmp=lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1]))
    # aa.sort(cmp=lambda a,b:int(a)-int(b))
    # aa.sort()
    for imgname in tqdm(aa):
        img = cv2.imread(os.path.join(imgdir,args.image_format%(imgname)))
        width, height = img.shape[1],img.shape[0]
        for pid in range(len(tracked[imgname])):
            pose = np.array(tracked[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
            tracked_id = tracked[imgname][pid]['idx']
            cv2.putText(img, \
                        text=str(tracked_id), \
                        org=(int(np.clip(pose[0,0],0,width)), int(np.clip(pose[0,1],0,height))), \
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                        fontScale=int(1*np.mean(pose[:,2])+1), \
                        color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), \
                        thickness=3)
            
            for idx_c in range(pose.shape[0]):
                cv2.circle(img,\
                    center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),\
                    # radius=int(20*np.mean(pose[:,2])+1),\
                    radius=3,\
                    color=(255,(80*idx_c)%255,0),\
                    thickness=-1)
            for idx in range(len(pairs)):
                cv2.line(img, \
                    pt1=(int(np.clip(pose[pairs[idx][0],0],0,width)),int(np.clip(pose[pairs[idx][0],1],0,height))), \
                    pt2=(int(np.clip(pose[pairs[idx][1],0],0,width)),int(np.clip(pose[pairs[idx][1],1],0,height))), \
                    # color=cmap(tracked_id*5), \
                    color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), \
                    # thickness=int(20*np.mean(pose[pairs[idx],2])+1)
                    thickness=3
                    )

        if not os.path.exists(visdir): 
            os.mkdir(visdir)
        # cv2.imwrite(os.path.join(visdir,str(int(imgname.split()[0])-min_frameNo)+".png"),img)
        cv2.imwrite(os.path.join(visdir,imgname+'.png'),img)
        out.write(img) # Write out frame to video
    print('demo image is generated in ',visdir)
    print('demo video is generated as: ',args.out_video_path)
    out.release()


def toFixNum_track(track_in,frame_list, max_pid_id_setting):
    for idx, frame_name in enumerate(tqdm(frame_list)):
        if track_in[frame_name]['num_boxes']<=max_pid_id_setting:
            continue
        ## make the first max_pid_id_setting persons  the person with max pose score.
        delete_count = track_in[frame_name]['num_boxes'] - max_pid_id_setting
        current_select_score = []
        current_select_score_pid_dict = {}
        for pid in range(1, max_pid_id_setting+1):
            current_select_score.append(track_in[frame_name][pid]['box_score'])
            current_select_score_pid_dict[track_in[frame_name][pid]['box_score']] = pid

        for pid in range(max_pid_id_setting+1, track_in[frame_name]['num_boxes']+1):
            current_min_score = min(current_select_score)
            if track_in[frame_name][pid]['box_score'] > min(current_select_score):
                min_score_pid = current_select_score_pid_dict[min(current_select_score)]
                track_in[frame_name][min_score_pid] = track_in[frame_name][pid]
                current_select_score_pid_dict[track_in[frame_name][pid]['box_score']] = min_score_pid
                current_select_score.remove(min(current_select_score))
                current_select_score.append(track_in[frame_name][pid]['box_score'])
                track_in[frame_name][pid] = None
                del track_in[frame_name][pid]

        track_in[frame_name]['num_boxes'] = max_pid_id_setting

    return track_in

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

parser = argparse.ArgumentParser(description='FoseFlow Tracker')
parser.add_argument('--imgdir', type=str, required=True, help="Must input the images dir")
parser.add_argument('--in_json', type=str, required=True, help="result json predicted by AlphaPose")
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

# super parameters
# 1. look-ahead LINK_LEN frames to find tracked human bbox
# 2. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score
# 3. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score(Non DeepMatching)
# 4. drop low-score(<DROP) keypoints
# 5. pick high-score(top NUM) keypoints when computing pose_IOU
# 6. box width/height around keypoint for computing pose IoU
# 7. match threshold in Hungarian Matching

link_len = args.link
# weights = [1,2,1,2,0,0] #dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score
# weights = [0,3,0,3,0,0] #dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score
weights = args.weights #dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score
# weights = [2,4,0,0,0,0] #dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score
# weights = [0,6,0,0,0,0] #dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score
weights_fff = [0,1,0,1,0,0]
drop = args.drop
num = args.num
mag = args.mag
match_thres = args.match

notrack_json = args.in_json
tracked_json = args.out_json
image_dir = args.imgdir
vis_dir = args.visdir

# if json format is differnt from "alphapose-forvis.json" (pytorch version)
if "forvis" not in notrack_json:
    results_forvis = {}
    last_image_name = ' '

    with open(notrack_json) as f:
        results = json.load(f)
        for i in xrange(len(results)):
            # imgpath = results[i]['image_id']
            imgpath = results[i]['file_name'].split('.')[0]

            # print(results[i])
            if last_image_name != imgpath:
                results_forvis[imgpath] = []
                results_forvis[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
            else:
                results_forvis[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
            last_image_name = imgpath
    notrack_json = os.path.join(os.path.dirname(notrack_json), "alphapose-results-forvis.json")
    with open(notrack_json,'w') as json_file:
            json_file.write(json.dumps(results_forvis))

notrack = {}
track = {}
num_persons = 0

# load json file without tracking information
print("\nStart loading json file...")
with open(notrack_json,'r') as f:
    notrack = json.load(f)
    if args.max_pid_id_setting!=-1:
        print('remove extract persons...')
        notrack = toFixNum_notrack(notrack,notrack.keys(), args.max_pid_id_setting)
    for img_name in tqdm(sorted(notrack.keys())):
        # print(img_name)
        track[img_name] = {'num_boxes':len(notrack[img_name])}
        for bid in range(len(notrack[img_name])):
            track[img_name][bid+1] = {}
            track[img_name][bid+1]['box_score'] = notrack[img_name][bid]['scores']
            track[img_name][bid+1]['box_pos'] = get_box(notrack[img_name][bid]['keypoints'], os.path.join(image_dir,args.image_format%(img_name)))
            track[img_name][bid+1]['box_pose_pos'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,0:2]
            track[img_name][bid+1]['box_pose_score'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,-1]


np.save('notrack-bl.npy',track)
# track = np.load('notrack-bl.npy').item()

# tracking process
max_pid_id = 0
frame_list = sorted(list(track.keys()))

tracker = Tracker(160, 30, 5, 100)


print("\nStart pose tracking...")
for idx, frame_name in enumerate(tqdm(frame_list[:-1])):
    frame_new_pids = []
    frame_id = frame_name.split(".")[0]

    next_frame_name = frame_list[idx+1]
    next_frame_id = next_frame_name.split(".")[0]

    # init tracking info of the first frame in one video
    if idx == 0:
        for pid in range(1, track[frame_name]['num_boxes']+1):
                track[frame_name][pid]['new_pid'] = pid
                track[frame_name][pid]['match_score'] = 0

    max_pid_id = max(max_pid_id, track[frame_name]['num_boxes'])
    cor_file = os.path.join(image_dir, "".join([frame_id, '_', next_frame_id, '_orb.txt']))

    # regenerate the missed pair-matching txt
    if not os.path.exists(cor_file) or os.stat(cor_file).st_size<200:
        img1_path = os.path.join(image_dir, args.image_format%(frame_name))
        img2_path = os.path.join(image_dir, args.image_format%(next_frame_name))

        orb_matching(img1_path,img2_path, image_dir, frame_id, next_frame_id)

    all_cors = np.loadtxt(cor_file)

    # if there is no people in this frame, then copy the info from former frame
    if track[next_frame_name]['num_boxes'] == 0:
        track[next_frame_name] = copy.deepcopy(track[frame_name])
        continue
    # # get all the newest people info from frame idx-linklen to idx
    # cur_all_pids, cur_all_pids_fff = stack_all_pids(track, frame_list[:-1], idx, max_pid_id, link_len)
    # match_indexes, match_scores = best_matching_hungarian(
    #     all_cors, cur_all_pids, cur_all_pids_fff, track[next_frame_name], weights, weights_fff, num, mag)
    # # match_indexes, match_scores = best_matching_boxIOU(cur_all_pids,cur_all_pids_fff)

    frame = cv2.imread(os.path.join(image_dir, args.image_format%(frame_name)))
    frame = kalman_matching(tracker,frame, track[next_frame_name])
    cv2.imwrite('./kalman_matching_vis/'+args.image_format%(frame_name),frame)


#     for pid1, pid2 in match_indexes:
#         if match_scores[pid1][pid2] > match_thres:
#             track[next_frame_name][pid2+1]['new_pid'] = cur_all_pids[pid1]['new_pid']
#             max_pid_id = max(max_pid_id, track[next_frame_name][pid2+1]['new_pid'])
#             track[next_frame_name][pid2+1]['match_score'] = match_scores[pid1][pid2]

#     # add the untracked new person
#     for next_pid in range(1, track[next_frame_name]['num_boxes'] + 1):
#         if 'new_pid' not in track[next_frame_name][next_pid]:
#             max_pid_id += 1
#             track[next_frame_name][next_pid]['new_pid'] = max_pid_id
#             track[next_frame_name][next_pid]['match_score'] = 0

# np.save('track-bl.npy',track)
# # track = np.load('track-bl.npy').item()

# # calculate number of people
# num_persons = 0
# for fid, frame_name in enumerate(frame_list):
#     for pid in range(1, track[frame_name]['num_boxes']+1):
#         num_persons = max(num_persons, track[frame_name][pid]['new_pid'])
# print("This video contains %d people."%(num_persons))

# # export tracking result into notrack json files
# print("\nExport tracking results to json...")
# for fid, frame_name in enumerate(tqdm(frame_list)):
#     for pid in range(track[frame_name]['num_boxes']):
#         notrack[frame_name][pid]['idx'] = track[frame_name][pid+1]['new_pid']

# with open(tracked_json,'w') as json_file:
#     json_file.write(json.dumps(notrack))

# if len(args.visdir)>0:
#     cmap = plt.cm.get_cmap("hsv", num_persons)
#     display_pose_cv2(image_dir, vis_dir, notrack, cmap, args)

os.system('rm {}/*.txt'.format(args.imgdir))

# os.system('rm ./test.mp4')
# os_code = os.system('ffmpeg -loop 1 -f image2 -i {}/%d.png -vcodec libx264 -r 10 -t 10 test.mp4'.format(args.visdir))
# if os_code==0:
#     print('./test.mp4 generated(image in {} is used)'.format(args.visdir))
# else:
#     print('./test.mp4 not generated, please run the following cmd manually to generate demo video')
#     print('ffmpeg -loop 1 -f image2 -i {}/%d.png -vcodec libx264 -r 10 -t 10 test.mp4'.format(args.visdir))
    

# print('images generated.')
# print('please refer to the following cmd to generate video:')
# print('ffmpeg -loop 1 -f image2 -i {}/%d.png -vcodec libx264 -r 10 -t 10 test.mp4'.format(args.visdir))
