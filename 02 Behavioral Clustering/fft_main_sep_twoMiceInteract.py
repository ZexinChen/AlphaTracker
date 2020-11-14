import numpy as np
import cv2
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import seaborn as sns
from time import time 
from sklearn.manifold import TSNE 
import umap
from utils_file import setting, contour_utils
import pickle
import fft_utils


############### Import Setting ###################
arg = setting.args_class()
arg.cluster_ids = [0]
arg.intervals = [[] for i in range(len(arg.tracked_json))]

############### Save Setting file ###################
##### copy the setting file to generated folder 
gen_video_folder = arg.gen_video_folder+'/'+arg.video_name_suffix+'/'
contour_utils.mkdir_p(gen_video_folder)
import os
cmd = 'cp ./fft_main_sep_twoMiceInteract.py %s/'%(gen_video_folder)
os.system(cmd)
cmd = 'cp ./utils_file/setting.py %s/'%(gen_video_folder)
os.system(cmd)



############### Load and Preprocess Data ###################
###  load pose and prepare images
arg, clips_dict = fft_utils.retrieve_poses_Mice(arg)

### filter clips that are too short 
clips_dict = fft_utils.clean_differentLength_clips(clips_dict)

### remove clips where two mice are too far away from each other
clips_dict = fft_utils.remove_longMiceDist_clips(arg,clips_dict)

### corret the left/right
clips_dict['pose_clips'] = fft_utils.left_right(clips_dict['pose_clips'])
clips_dict['poseTheOther_clips'] = fft_utils.left_right(clips_dict['poseTheOther_clips'])

### remove limbs that is not correct
clips_dict['pose_clips'] = fft_utils.correctLimbs(clips_dict['pose_clips'])
clips_dict['poseTheOther_clips'] = fft_utils.correctLimbs(clips_dict['poseTheOther_clips'])

### align Pose
clips_dict['pose_clips_align'] = fft_utils.align_poses_self(clips_dict['pose_clips'])
clips_dict['poseTheOther_clips_alignSelf'] = fft_utils.align_poses_self(clips_dict['poseTheOther_clips'])
clips_dict['poseTheOther_clips_alignToOther'] = fft_utils.align_poses_toTheFirst(clips_dict['pose_clips'],clips_dict['poseTheOther_clips'])

### generate features 
feature_clips_dict,all_info_selected,raw_feature_clips_dict = \
        fft_utils.compute_features_sep_twoMice_Independent(arg,clips_dict)


################ Cluster  ##################
# all_result = []
# all_result_dict = {}
# for c_arg_dict_idx in range(len(arg.cluster_arg)):
#     c_arg_dict = arg.cluster_arg[c_arg_dict_idx]
#     print('clustering with %s, with thred:%f'%(c_arg_dict['name'],c_arg_dict['thred']))
#     ### Normalize and concatenate feature  
#     feat_norm = c_arg_dict['features_arg'][0]['weight']*fft_utils.get_normFeature_byArg(c_arg_dict['features_arg'][0],feature_clips_dict)
#     for f_i in range(1,len(c_arg_dict['features_arg'])):
#         feat_norm_tmp = c_arg_dict['features_arg'][f_i]['weight']*fft_utils.get_normFeature_byArg(c_arg_dict['features_arg'][f_i],feature_clips_dict)
#         feat_norm = np.concatenate([feat_norm,feat_norm_tmp] ,axis = 1)
        
#     ### hierarchical clustering with the selected features
#     feature_clips_forCluster = feat_norm
#     clusters,Z,cluster_result_fcluster,leaves = fft_utils.cluster_bak1(c_arg_dict,feature_clips_forCluster)
#     print('number of clusters:%d'%(np.max(cluster_result_fcluster)))  
    
#     ### t sne 
#     if arg.DR_method == 'tsne':
#         n_components = 2 
#         tsne = TSNE(n_components=n_components, init='pca', random_state=0)
#         Y = tsne.fit_transform(feat_norm)
#     elif arg.DR_method == 'umap':
#         um = umap.UMAP(n_neighbors=5,
#                       min_dist=0.3,
#                       metric='correlation')
#         Y = um.fit_transform(feat_norm)
#     else:
#         print('dimension reduction algorithm %s is not defined'%(arg.DR_method))
#         raise
    
#     all_result.append({'clusters':clusters,\
#                        'Z':Z,\
#                        'cluster_result_fcluster':cluster_result_fcluster,\
#                        'leaves':leaves,\
#                        'dimension_reduction_Y':Y
#                       })
#     all_result_dict[c_arg_dict['name']]= {'clusters':clusters,\
#                        'Z':Z,\
#                        'cluster_result_fcluster':cluster_result_fcluster,\
#                        'leaves':leaves,\
#                        'dimension_reduction_Y':Y
#                       }

all_result,all_result_dict = fft_utils.cluster(arg,feature_clips_dict,all_info_selected,raw_feature_clips_dict)


# ### save file for UI
# fft_utils.write_cluster_result_to_infoDict(all_result,all_info_selected)
# fft_utils.save_Z_and_clips(arg,all_result_dict,all_info_selected)

### merge seperate cluster 
name = [arg.cluster_arg[c_arg_dict_i]['name'] for c_arg_dict_i in range(len(arg.cluster_arg))]

cluster_merged_dict = {}
print('clustering using:',[arg.cluster_arg[c_arg_dict_i]['name'] for c_arg_dict_i in arg.cluster_ids])
for data_i in range(len(all_result[0]['cluster_result_fcluster'])):
    cluster_key = ''
    for c_i in arg.cluster_ids:
        cluster_key = cluster_key + str(all_result[c_i]['cluster_result_fcluster'][data_i]) + '_'
    cluster_key = cluster_key[:-1]
    if cluster_key in cluster_merged_dict:
        cluster_merged_dict[cluster_key].append(data_i)
    else:
        cluster_merged_dict[cluster_key] = [data_i]

with open(arg.result_folder + '/feature_clips_dict.pckl','wb') as f:
     pickle.dump(feature_clips_dict,f)

with open(arg.result_folder + '/all_info_selected.pckl','wb') as f:
     pickle.dump(all_info_selected,f)

with open(arg.result_folder +'/raw_feature_clips_dict.pckl','wb') as f:
     pickle.dump(raw_feature_clips_dict,f)

with open(arg.result_folder + '/all_result.pckl','wb') as f:
     pickle.dump(all_result,f)

with open(arg.result_folder + '/all_result_dict.pckl','wb') as f:
     pickle.dump(all_result_dict,f)
        
        
################ visualize  ##################
fft_utils.visualize_inSepVideo_twoMice(arg,\
    arg.cluster_ids,name,cluster_merged_dict,
    all_result, \
    all_info_selected['pose_clips'],\
    all_info_selected['poseTheOther_clips'],\
    all_info_selected['frames_path_clips'],\
    all_info_selected['cont_clips'],raw_feature_clips_dict, feature_clips_dict, \
    video_name_suffix=arg.video_name_suffix,gen_video_folder=gen_video_folder,max_clips_num=arg.max_clips_num)








