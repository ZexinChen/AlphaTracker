import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import seaborn as sns
from time import time 
from sklearn.manifold import TSNE 
from fft_utils import *


############### Setting ###################
import fft_utils
from utils_file import setting, contour_utils
# arg = fft_utils.args_class()
arg = setting.args_class()

# ############### Generate Pose ###################
# arg,pose_clips,info_clips,cont_clips = fft_utils.retrieve_poses(arg)

# pose_clips = np.asarray(pose_clips)
# cont_clips = np.asarray(cont_clips)


import pickle
with open(arg.result_folder + '/feature_clips_dict.pckl','rb') as f:
    feature_clips_dict = pickle.load(f)
with open(arg.result_folder + '/all_result.pckl','rb') as f:
    all_result = pickle.load(f)

for c_arg_dict_idx in range(len(arg.cluster_arg)):
    c_arg_dict = arg.cluster_arg[c_arg_dict_idx]
    print('clustering with {}, with thred:{}'.format(c_arg_dict['name'],c_arg_dict['thred']))
    ### Normalize and concatenate feature  
    feat_norm = c_arg_dict['features_arg'][0]['weight']*get_normFeature_byArg(c_arg_dict['features_arg'][0],feature_clips_dict)
    for f_i in range(1,len(c_arg_dict['features_arg'])):
        feat_norm_tmp = c_arg_dict['features_arg'][f_i]['weight']*get_normFeature_byArg(c_arg_dict['features_arg'][f_i],feature_clips_dict)
        feat_norm = np.concatenate([feat_norm,feat_norm_tmp] ,axis = 1)

    clip_num,fea_num = feat_norm.shape
    print(clip_num,fea_num)

def dist(a,b):
    return np.linalg.norm(a-b)

number_of_example_clip = 3
cluster_result_fcluster = all_result[0]['cluster_result_fcluster']
for cluster_id in range(np.max(cluster_result_fcluster)):
    distance_sum = [0 for iii in range(len(cluster_result_fcluster))]
    print('\r processing the %d cluster'%cluster_id,end='')
    for i in range(clip_num):
        if cluster_result_fcluster[i] == cluster_id:
            for j in range(clip_num):
                if cluster_result_fcluster[j] == cluster_id:
                    distance_sum[i] += dist(feat_norm[i,:],feat_norm[j,:])
    print(distance_sum)
    distance_sum = np.asarray(distance_sum)
    distance_sum[distance_sum==0] = max(distance_sum) + 1
    index_of_first_n = distance_sum.argsort()[:number_of_example_clip]
    print(distance_sum)
    print('Clip No of the example clips for cluster {}: {}'.format(cluster_id,index_of_first_n))

    
    



