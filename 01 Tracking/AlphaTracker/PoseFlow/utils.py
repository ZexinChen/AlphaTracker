# coding: utf-8

'''
File: utils.py
Project: AlphaPose
File Created: Thursday, 1st March 2018 5:32:34 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 20th March 2018 1:18:17 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import numpy as np
import cv2 as cv
import os
import json
import copy
import heapq
from munkres import Munkres, print_matrix
from PIL import Image
from tqdm import tqdm
import cv2


# keypoint penalty weight
delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891, 0.01402144, \
                    0.03909642, 0.03686941, 0.01981803, 0.03843971, 0.03412318, 0.02415081, \
                    0.01291456, 0.01236173,0.01291456, 0.01236173])

track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

# get expand bbox surrounding single person's keypoints
def get_box(pose, imgpath):

    pose = np.array(pose).reshape(-1,3)
    xmin = np.min(pose[:,0])
    xmax = np.max(pose[:,0])
    ymin = np.min(pose[:,1])
    ymax = np.max(pose[:,1])
    
    img_height, img_width, _ = cv.imread(imgpath).shape

    return expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height)

# def select_box(pose,boxes):
#     pose = np.array(pose).reshape(-1,3)
#     xmin = np.min(pose[:,0])
#     xmax = np.max(pose[:,0])
#     ymin = np.min(pose[:,1])
#     ymax = np.max(pose[:,1])

#     for b in boxes:#d['x'], d['y'], d['x']+d['width'], d['y']+d['height']
#         if xmin>= b[0] and 


# expand bbox for containing more background
def expand_bbox(left, right, top, bottom, img_width, img_height):

    width = right - left
    height = bottom - top
    ratio = 0.1 # expand ratio
    new_left = np.clip(left - ratio * width, 0, img_width)
    new_right = np.clip(right + ratio * width, 0, img_width)
    new_top = np.clip(top - ratio * height, 0, img_height)
    new_bottom = np.clip(bottom + ratio * height, 0, img_height)

    return [int(new_left), int(new_right), int(new_top), int(new_bottom)]

# calculate final matching grade
def cal_grade(l, w):
    return sum(np.array(l)*np.array(w))

# calculate IoU of two boxes(thanks @ZongweiZhou1)
def cal_bbox_iou(boxA, boxB): 

    xA = max(boxA[0], boxB[0]) #xmin
    yA = max(boxA[2], boxB[2]) #ymin
    xB = min(boxA[1], boxB[1]) #xmax
    yB = min(boxA[3], boxB[3]) #ymax

    if xA < xB and yA < yB: 
        interArea = (xB - xA + 1) * (yB - yA + 1) 
        boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1) 
        boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1) 
        iou = interArea / float(boxAArea + boxBArea - interArea+0.00001) 
    else: 
        iou=0.0

    return iou

# calculate OKS between two single poses
def compute_oks(anno, predict, delta):
    
    xmax = np.max(np.vstack((anno[:, 0], predict[:, 0])))
    xmin = np.min(np.vstack((anno[:, 0], predict[:, 0])))
    ymax = np.max(np.vstack((anno[:, 1], predict[:, 1])))
    ymin = np.min(np.vstack((anno[:, 1], predict[:, 1])))
    scale = (xmax - xmin) * (ymax - ymin)
    dis = np.sum((anno - predict)**2, axis=1)
    oks = np.mean(np.exp(-dis / 2 / delta**2 / scale))

    return oks

# stack all already tracked people's info together(thanks @ZongweiZhou1)
def stack_all_pids(track_vid, frame_list, idxs, max_pid_id, link_len):
    
    #track_vid contains track_vid[<=idx]
    all_pids_info = []
    all_pids_fff = [] # boolean list, 'fff' means From Former Frame
    all_pids_ids = [(item+1) for item in range(max_pid_id)]
    
    for idx in np.arange(idxs,max(idxs-link_len,-1),-1):
        # print('!!!',track_vid[frame_list[idx]])
        for pid in range(1, track_vid[frame_list[idx]]['num_boxes']+1):
            if len(all_pids_ids) == 0:
                return all_pids_info, all_pids_fff
            elif track_vid[frame_list[idx]][pid]['new_pid'] in all_pids_ids:
                all_pids_ids.remove(track_vid[frame_list[idx]][pid]['new_pid'])
                all_pids_info.append(track_vid[frame_list[idx]][pid])
                if idx == idxs:
                    all_pids_fff.append(True)
                else:
                    all_pids_fff.append(False)
    return all_pids_info, all_pids_fff

# calculate DeepMatching Pose IoU given two boxes
def find_two_pose_box_iou(pose1_box, pose2_box, all_cors):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = pose1_box
    x1_region_ids = set(np.where((x1 >= x_min) & (x1 <= x_max))[0].tolist())
    y1_region_ids = set(np.where((y1 >= y_min) & (y1 <= y_max))[0].tolist())
    region_ids1 = x1_region_ids & y1_region_ids
    x_min, x_max, y_min, y_max = pose2_box
    x2_region_ids = set(np.where((x2 >= x_min) & (x2 <= x_max))[0].tolist())
    y2_region_ids = set(np.where((y2 >= y_min) & (y2 <= y_max))[0].tolist())
    region_ids2 = x2_region_ids & y2_region_ids
    inter = region_ids1 & region_ids2
    union = region_ids1 | region_ids2
    pose_box_iou = len(inter) / (len(union) + 0.00001)

    return pose_box_iou

# calculate general Pose IoU(only consider top NUM matched keypoints)
def cal_pose_iou(pose1_box,pose2_box, num,mag):
    
    pose_iou = []
    for row in range(len(pose1_box)):
        x1,y1 = pose1_box[row]
        x2,y2 = pose2_box[row]
        box1 = [x1-mag,x1+mag,y1-mag,y1+mag]
        box2 = [x2-mag,x2+mag,y2-mag,y2+mag]
        pose_iou.append(cal_bbox_iou(box1,box2))

    return np.mean(heapq.nlargest(num, pose_iou))

# calculate DeepMatching based Pose IoU(only consider top NUM matched keypoints)
def cal_pose_iou_dm(all_cors,pose1,pose2,num,mag):
    
    poses_iou = []
    for ids in range(len(pose1)):
        pose1_box = [pose1[ids][0]-mag,pose1[ids][0]+mag,pose1[ids][1]-mag,pose1[ids][1]+mag]
        pose2_box = [pose2[ids][0]-mag,pose2[ids][0]+mag,pose2[ids][1]-mag,pose2[ids][1]+mag]
        poses_iou.append(find_two_pose_box_iou(pose1_box, pose2_box, all_cors))

    return np.mean(heapq.nlargest(num, poses_iou))
        
def select_max(cost_matrix):
    cost_matrix_copy = copy.deepcopy(cost_matrix)
    selectIdx = []
    for ii in range(cost_matrix_copy.shape[1]):
        xs,ys = np.where(cost_matrix_copy==np.max(cost_matrix_copy))
        selectIdx.append((xs[0],ys[0]))
        cost_matrix_copy[xs[0],:] = 0
        cost_matrix_copy[:,ys[0]] = 0

    return selectIdx
    

# hungarian matching algorithm(thanks @ZongweiZhou1)
def best_matching_hungarian_newselect(all_cors, all_pids_info, all_pids_fff, track_vid_next_fid, weights, weights_fff, num, mag):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    all_grades_details = []
    all_grades = []
    
    box1_num = len(all_pids_info)
    box2_num = track_vid_next_fid['num_boxes']
    cost_matrix = np.zeros((box1_num, box2_num))

    for pid1 in range(box1_num):
        box1_pos = all_pids_info[pid1]['box_pos']
        box1_region_ids = find_region_cors_last(box1_pos, all_cors)
        box1_score = all_pids_info[pid1]['box_score']
        box1_pose = all_pids_info[pid1]['box_pose_pos']
        box1_fff = all_pids_fff[pid1]

        for pid2 in range(1, track_vid_next_fid['num_boxes'] + 1):
            box2_pos = track_vid_next_fid[pid2]['box_pos']
            box2_region_ids = find_region_cors_next(box2_pos, all_cors)
            box2_score = track_vid_next_fid[pid2]['box_score']
            box2_pose = track_vid_next_fid[pid2]['box_pose_pos']
                        
            inter = box1_region_ids & box2_region_ids
            union = box1_region_ids | box2_region_ids
            dm_iou = len(inter) / (len(union) + 0.00001)
            box_iou = cal_bbox_iou(box1_pos, box2_pos)
            pose_iou_dm = cal_pose_iou_dm(all_cors, box1_pose, box2_pose, num,mag)
            pose_iou = cal_pose_iou(box1_pose, box2_pose,num,mag)
            if box1_fff:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights)
            else:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights_fff)
                
            cost_matrix[pid1, pid2 - 1] = grade
    indexes = select_max(cost_matrix)
    # m = Munkres()
    # indexes = m.compute((-np.array(cost_matrix)).tolist())

    return indexes, cost_matrix

def best_matching_hungarian(all_cors, all_pids_info, all_pids_fff, track_vid_next_fid, weights, weights_fff, num, mag):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    all_grades_details = []
    all_grades = []
    
    box1_num = len(all_pids_info)
    box2_num = track_vid_next_fid['num_boxes']
    cost_matrix = np.zeros((box1_num, box2_num))

    for pid1 in range(box1_num):
        box1_pos = all_pids_info[pid1]['box_pos']
        box1_region_ids = find_region_cors_last(box1_pos, all_cors)
        box1_score = all_pids_info[pid1]['box_score']
        box1_pose = all_pids_info[pid1]['box_pose_pos']
        box1_fff = all_pids_fff[pid1]

        for pid2 in range(1, track_vid_next_fid['num_boxes'] + 1):
            box2_pos = track_vid_next_fid[pid2]['box_pos']
            box2_region_ids = find_region_cors_next(box2_pos, all_cors)
            box2_score = track_vid_next_fid[pid2]['box_score']
            box2_pose = track_vid_next_fid[pid2]['box_pose_pos']
                        
            inter = box1_region_ids & box2_region_ids
            union = box1_region_ids | box2_region_ids
            dm_iou = len(inter) / (len(union) + 0.00001)
            box_iou = cal_bbox_iou(box1_pos, box2_pos)
            pose_iou_dm = cal_pose_iou_dm(all_cors, box1_pose, box2_pose, num,mag)
            pose_iou = cal_pose_iou(box1_pose, box2_pose,num,mag)
            if box1_fff:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights)
            else:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights_fff)
                
            cost_matrix[pid1, pid2 - 1] = grade
    # indexes = select_max(cost_matrix)
    m = Munkres()
    indexes = m.compute((-np.array(cost_matrix)).tolist())

    return indexes, cost_matrix

def best_matching_hungarian_noORB(all_cors, all_pids_info, all_pids_fff, track_vid_next_fid, weights, weights_fff, num, mag):
    
    # x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    all_grades_details = []
    all_grades = []
    
    box1_num = len(all_pids_info)
    box2_num = track_vid_next_fid['num_boxes']
    cost_matrix = np.zeros((box1_num, box2_num))

    for pid1 in range(box1_num):
        box1_pos = all_pids_info[pid1]['box_pos']
        # box1_region_ids = find_region_cors_last(box1_pos, all_cors)
        box1_score = all_pids_info[pid1]['box_score']
        box1_pose = all_pids_info[pid1]['box_pose_pos']
        box1_fff = all_pids_fff[pid1]

        for pid2 in range(1, track_vid_next_fid['num_boxes'] + 1):
            box2_pos = track_vid_next_fid[pid2]['box_pos']
            # box2_region_ids = find_region_cors_next(box2_pos, all_cors)
            box2_score = track_vid_next_fid[pid2]['box_score']
            box2_pose = track_vid_next_fid[pid2]['box_pose_pos']
                        
            # inter = box1_region_ids & box2_region_ids
            # union = box1_region_ids | box2_region_ids
            # dm_iou = len(inter) / (len(union) + 0.00001)
            dm_iou = 0
            box_iou = cal_bbox_iou(box1_pos, box2_pos)
            # pose_iou_dm = cal_pose_iou_dm(all_cors, box1_pose, box2_pose, num,mag)
            pose_iou_dm = 0
            pose_iou = cal_pose_iou(box1_pose, box2_pose,num,mag)
            if box1_fff:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights)
            else:
                grade = cal_grade([dm_iou, box_iou, pose_iou_dm, pose_iou, box1_score, box2_score], weights_fff)
                
            cost_matrix[pid1, pid2 - 1] = grade
    indexes = select_max(cost_matrix)
    # m = Munkres()
    # indexes = m.compute((-np.array(cost_matrix)).tolist())

    return indexes, cost_matrix

def kalman_matching(tracker,frame, track_vid_next_fid):
    all_boxCenter = []
    for pid2 in range(1, track_vid_next_fid['num_boxes'] + 1):
        box2_pos = track_vid_next_fid[pid2]['box_pos']
        [left, right, top, bottom] = box2_pos
        all_boxCenter.append([(left+right)/2,(top+bottom)/2])
        cv2.circle(frame,(int((left+right)/2), int((top+bottom)/2)),5,(255,(100*pid2)%255,0),-1)
    tracker.Update(all_boxCenter)
    for i in range(len(tracker.tracks)):
        if (len(tracker.tracks[i].trace) > 1):
            for j in range(len(tracker.tracks[i].trace)-1):
                # Draw trace line
                x1 = tracker.tracks[i].trace[j][0][0]
                y1 = tracker.tracks[i].trace[j][1][0]
                x2 = tracker.tracks[i].trace[j+1][0][0]
                y2 = tracker.tracks[i].trace[j+1][1][0]
                clr = tracker.tracks[i].track_id % 9
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         track_colors[clr], 2)
    return frame



# calculate number of matching points in one box from last frame
def find_region_cors_last(box_pos, all_cors):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = box_pos
    x1_region_ids = set(np.where((x1 >= x_min) & (x1 <= x_max))[0].tolist())
    y1_region_ids = set(np.where((y1 >= y_min) & (y1 <= y_max))[0].tolist())
    region_ids = x1_region_ids & y1_region_ids

    return region_ids

# calculate number of matching points in one box from next frame
def find_region_cors_next(box_pos, all_cors):
    
    x1, y1, x2, y2 = [all_cors[:, col] for col in range(4)]
    x_min, x_max, y_min, y_max = box_pos
    x2_region_ids = set(np.where((x2 >= x_min) & (x2 <= x_max))[0].tolist())
    y2_region_ids = set(np.where((y2 >= y_min) & (y2 <= y_max))[0].tolist())
    region_ids = x2_region_ids & y2_region_ids

    return region_ids

# fill the nose keypoint by averaging head and neck
def add_nose(array):
    
    if min(array.shape) == 2:
        head = array[-1,:]
        neck = array[-2,:]
    else:
        head = array[-1]
        neck = array[-2]
    nose = (head+neck)/2.0

    return np.insert(array,-1,nose,axis=0)

# list remove operation
def remove_list(l1,vname,l2):
    
    for item in l2:
        l1.remove(os.path.join(vname,item))
        
    return l1


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

## kalman
def kalman_example_using_lib():
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise

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

    dt = .1
    x = np.array([0., 0.]) 
    kf = pos_vel_filter(x, P=500, R=5, Q=0.1, dt=dt)

    from kf_book.mkf_internal import plot_track

    def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0, 
            track=None, zs=None,
            count=0, do_plot=True, **kwargs):
        """
        track is the actual position of the dog, zs are the 
        corresponding measurements. 
        """

        # Simulate dog if no data provided. 
        if zs is None:
            track, zs = compute_dog_data(R, Q, count)

        # create the Kalman filter
        kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)  

        # run the kalman filter and store the results
        xs, cov = [], []
        for z in zs:
            kf.predict()
            kf.update(z)
            xs.append(kf.x)
            cov.append(kf.P)

        xs, cov = np.array(xs), np.array(cov)
        if do_plot:
            plot_track(xs[:, 0], track, zs, cov, 
                       dt=dt, **kwargs)
        return xs, cov

    # from filterpy.common import Saver
    # s = Saver(kf)
    # for i in range(1, 6):
    #     kf.predict()
    #     kf.update([i])
    #     s.save()  # save the current state

def kalman_example_using_nolib():
    dt = 1.
    R_var = 10
    Q_var = 0.01
    x = np.array([[10.0, 4.5]]).T
    P = np.diag([500, 49])
    F = np.array([[1, dt],
                  [0,  1]])
    H = np.array([[1., 0.]])
    R = np.array([[R_var]])
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)

    from numpy import dot
    from scipy.linalg import inv

    count = 50
    track, zs = compute_dog_data(R_var, Q_var, count)
    xs, cov = [], []
    for z in zs:
        # predict
        x = dot(F, x)
        P = dot(F, P).dot(F.T) + Q
        
        #update
        S = dot(H, P).dot(H.T) + R
        K = dot(P, H.T).dot(inv(S))
        y = z - dot(H, x)
        x += dot(K, y)
        P = P - dot(K, H).dot(P)
        
        xs.append(x)
        cov.append(P)

    xs, cov = np.array(xs), np.array(cov)
    plot_track(xs[:, 0], track, zs, cov, plot_P=False, dt=dt)



def display_pose_cv2(imgdir, visdir, track_forJson, cmap, args):

    print("Start visualization...\n")
    colors =['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
    part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    pairs = [[0,1],[0,2],[0,3]]
    
    img_tmp = cv2.imread(os.path.join(imgdir,args.image_format%(list(track_forJson.keys())[0])))
    height, width, channels = img_tmp.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#     out = cv2.VideoWriter(args.out_video_path, fourcc, 20.0, (width, height))
   
    aa = list(track_forJson.keys())
    aa.sort(key=cmp_to_key(lambda a,b:int(a.split('_')[-1])-int(b.split('_')[-1])))
    for imgname in tqdm(aa[160:180]):
        img = cv2.imread(os.path.join(imgdir,args.image_format%(imgname)))
        width, height = img.shape[1],img.shape[0]
        cv2.putText(img, text=imgname,  org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=3)
        
        frame_info_list_pred = kalman_get(kfs,track_forJson,imgname)

        for pid in range(len(track_forJson[imgname])):
            if not track_forJson[imgname][pid]:
                continue
            pose = np.array(track_forJson[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
            joint_thred = 0.0
            tracked_id = track_forJson[imgname][pid]['idx']
            cv2.putText(img,  text=str(tracked_id), org=(int(np.clip(pose[0,0],0,width)), int(np.clip(pose[0,1],0,height))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=int(1*np.mean(pose[:,2])+1), color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), thickness=3)
            
            ## draw output of the neuron network
            for idx_c in range(pose.shape[0]):
                if(pose[idx_c,2]<joint_thred):
                    continue
                cv2.circle(img,center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),radius=6,color=((50*idx_c)%255,(80*idx_c)%255,(120*idx_c)%255),thickness=-1)
                cv2.putText(img, text='%0.2f'%(pose[idx_c,2]), org=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), thickness=2)
            
            for idx in range(len(pairs)):
                if(pose[pairs[idx][0],2]<joint_thred or pose[pairs[idx][1],2]<joint_thred):
                    continue
                cv2.line(img, pt1=(int(np.clip(pose[pairs[idx][0],0],0,width)),int(np.clip(pose[pairs[idx][0],1],0,height))), pt2=(int(np.clip(pose[pairs[idx][1],0],0,width)),int(np.clip(pose[pairs[idx][1],1],0,height))), color=((160*tracked_id)%255,(80*tracked_id)%255,(30*tracked_id)%255), thickness=3)

            ## draw output of the kalman filter
            pose = np.array(frame_info_list_pred[pid]['keypoints']).reshape(-1,3)[:,:3]
            for idx_c in range(pose.shape[0]):
                if(pose[idx_c,2]<joint_thred):
                    continue
                cv2.circle(img,center=(int(np.clip(pose[idx_c,0],0,width)), int(np.clip(pose[idx_c,1],0,height))),radius=6,color=(0,(80*idx_c)%255,(120*idx_c)%255),thickness=-1)
            

        if not os.path.exists(visdir): 
            os.mkdir(visdir)
        visImage(img)
#         out.write(img) # Write out frame to video
#     out.release()
    print('demo image is generated in ',visdir)
    print('demo video is generated as: ',args.out_video_path)









