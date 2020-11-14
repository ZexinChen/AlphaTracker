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


from utils_kalman import *


class arg_class(object):
    """docstring for arg_class"""
    def __init__(self):
        super(arg_class, self).__init__()
        self.link = 100
        self.drop = 2.0
        self.num = 7
        self.mag = 30
        self.match = 0.2


        self.imgdir = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/withLimbs_interaction_refine//oriFrameFromVideo//2019-11-19_2femalespost8hrisolation/frame_folder/'
        self.in_json = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/withLimbs_interaction_refine//alphapose-results.json'
        self.out_json = ' /disk1/zexin/project/mice/clustering_sequencial/track_result_folder/withLimbs_interaction_refine//alphapose-results-forvis-tracked.json'
        self.visdir = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/withLimbs_interaction_refine//pose_track_vis/'
        self.out_video_path = '/disk1/zexin/project/mice/clustering_sequencial/track_result_folder/withLimbs_interaction_refine//labeled_byCompany_0204050607_split90_ori_2_0_060000_tmp.mp4'        

        self.image_format = '%s.png'
        self.max_pid_id_setting = 2 
        self.num_pose = 4
        self.match = 0  
        self.weights = [0, 6, 0, 0, 0, 0,]  
        
        self.fill_blank_with_predict = False
        self.kalman = True
        self.smooth_pose = True
    
args = arg_class()


link_len = args.link
weights = args.weights #dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score
weights_fff = args.weights
drop = args.drop
num = args.num
mag = args.mag
match_thres = args.match

notrack_json = args.in_json
tracked_json = args.out_json
image_dir = args.imgdir
vis_dir = args.visdir


if "forvis" not in notrack_json:
    results_forvis = {}
    last_image_name = ' '

    with open(notrack_json) as f:
        results = json.load(f)
        for i in range(len(results)):
            # imgpath = results[i]['image_id']
            imgpath = results[i]['file_name'].split('.')[0]

            # print(results[i])
            if last_image_name != imgpath:
                results_forvis[imgpath] = []
                results_forvis[imgpath].append(\
                    {'keypoints':results[i]['keypoints'],\
                    'scores':results[i]['score'],\
                    'box':results[i]['box']})
            else:
                results_forvis[imgpath].append(\
                    {'keypoints':results[i]['keypoints'],\
                    'scores':results[i]['score'],\
                    'box':results[i]['box']})
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
            # track[img_name][bid+1]['box_pos'] = get_box(notrack[img_name][bid]['keypoints'], os.path.join(image_dir,args.image_format%(img_name)))
            track[img_name][bid+1]['box_pos'] =  [ int(notrack[img_name][bid]['box'][0]),\
                                                   int(notrack[img_name][bid]['box'][2]),\
                                                   int(notrack[img_name][bid]['box'][1]),\
                                                   int(notrack[img_name][bid]['box'][3])]
                                                   #left top right bottom   to left right top bottom
            track[img_name][bid+1]['box_pose_pos'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,0:2]
            track[img_name][bid+1]['box_pose_score'] = np.array(notrack[img_name][bid]['keypoints']).reshape(-1,3)[:,-1]

np.save('notrack-bl.npy',track)
# track = np.load('notrack-bl.npy').item()


# tracking process
max_pid_id = 0
frame_list = list(track.keys())
frame_list.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))

print("\nStart pose tracking...")
for idx, frame_name in enumerate(tqdm(frame_list[:-1])):
    start_time = time.time()
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
    
    # if there is no people in this frame, then copy the info from former frame
    if track[next_frame_name]['num_boxes'] == 0:
        track[next_frame_name] = copy.deepcopy(track[frame_name])
        continue

    # get all the newest people info from frame idx-linklen to idx
    cur_all_pids, cur_all_pids_fff = stack_all_pids(track, \
        frame_list[:-1], \
        idx, max_pid_id, link_len)
    stack_time = time.time()

    match_indexes, match_scores = best_matching_hungarian_noORB(
        None, cur_all_pids, cur_all_pids_fff, \
        track[next_frame_name], weights, weights_fff, num, mag)
    match_time = time.time()


    pid_remain = [i+1 for i in range(args.max_pid_id_setting)]
    pid_conflict = []
    pid2s_checked = []
    pid1s_checked = []

    for pid1, pid2 in match_indexes:
        if match_scores[pid1][pid2] > match_thres:

            if args.max_pid_id_setting!=-1:
                if pid2 in pid2s_checked or pid1 in pid1s_checked:
                    pid_conflict.append([pid1,pid2])
                    continue
                else:
                    pid_remain.remove(cur_all_pids[pid1]['new_pid'])
                    pid2s_checked.append(pid2)
                    pid1s_checked.append(pid1)

            track[next_frame_name][pid2+1]['new_pid'] = cur_all_pids[pid1]['new_pid']
            track[next_frame_name][pid2+1]['match_score'] = match_scores[pid1][pid2]


            if track[next_frame_name][pid2+1]['new_pid']>max_pid_id:
                print('tracking warning!!\n track[next_frame_name][pid2+1][\'new_pid\']>max_pid_id:',next_frame_name,track[next_frame_name][pid2+1]['new_pid'],max_pid_id)


    # add the untracked new person
    for next_pid in range(1, track[next_frame_name]['num_boxes'] + 1):
        if 'new_pid' not in track[next_frame_name][next_pid]:
            if args.max_pid_id_setting!=-1:
                if len(pid_remain)!=0:
                    track[next_frame_name][next_pid]['new_pid'] = pid_remain[0]
                    del pid_remain[0]
            else:
                max_pid_id += 1
                track[next_frame_name][next_pid]['new_pid'] = max_pid_id
                track[next_frame_name][next_pid]['match_score'] = 0

np.save('track-bl.npy',track)


# calculate number of people
num_persons = 0
for fid, frame_name in enumerate(frame_list):
    for pid in range(1, track[frame_name]['num_boxes']+1):
        num_persons = max(num_persons, track[frame_name][pid]['new_pid'])
print("This video contains %d people."%(num_persons))

# export tracking result into notrack json files
print("\nExport tracking results to json...")
for fid, frame_name in enumerate(tqdm(frame_list)):
    for pid in range(track[frame_name]['num_boxes']):
        notrack[frame_name][pid]['idx'] = track[frame_name][pid+1]['new_pid']

# export tracking result into new json files
track_forJson = {}
for fid, frame_name in enumerate(tqdm(frame_list)):
    
    track_forJson[frame_name] = [{} for pid in range(num_persons) ]
    for pid in range(track[frame_name]['num_boxes']):
        idx_in_trackedJson = track[frame_name][pid+1]['new_pid']-1
        try:
            track_forJson[frame_name][idx_in_trackedJson] = notrack[frame_name][pid]
        except Exception as e:
            print(track_forJson[frame_name])
            print(notrack[frame_name])
            print(idx_in_trackedJson)
            print(pid)
            raise e



# kfs = init_kfs(track_forJson,args.max_pid_id_setting,args.num_pose,init_id=140)    

# if len(args.visdir)>0:
#     cmap = plt.cm.get_cmap("hsv", num_persons)
# #     display_pose_cv2(image_dir, vis_dir, notrack, cmap, args)
#     display_pose_cv2(image_dir, vis_dir, track_forJson,kfs, cmap, args)

# track_forJson = copy.deepcopy(track_forJson_save)
post_process_tracking(track_forJson,args)
display_track_forJson(track_forJson,args)
















