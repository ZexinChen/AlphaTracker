# Behavioral Clustering

### Algorithm overview

The main process of hierarchical clustering list below can be found in 	`./fft_main_sep_twoMiceInteract.py`
1. Get data from tracking result with function `retrieve_poses_Mice`
2. Clean the data with functions `clean_differentLength_clips`, `remove_longMiceDist_clips`, `left_right`, `correctLimbs`
3. Preprocess the data with function `align_poses_self`
4. Get feature from the data with funcion `compute_features_sep_twoMice_Independent`
5. Conduct hierarchical clustering with function `cluster`
6. Visualize and save the result  
7. Generate input files for clustering UI

<br>

## Installation

See the installation for the tracking part.

<br>

## Run clustering algorithm

### Step 1. Configuration

Set the Behavioral Clustering folder as the current directory.

``` 
cd "02 Behavioral Clustering"
```

Change the following settings in ```./utils_file/setting.py```. Changes for individual behavior and social behavior are explained separately here.

#### Configuration for individual behavior

- Adjust the weights for Individual behavioral features

```
        # cluster_arg: parameter of features and threshold for clustering
        self.cluster_arg = [
        {
            'thred':30, # threshold for defining cluster in the dendrogram
            'name':'all_twoMice', ## name: for display
            'evaluation_metric':'Adjusted Rand index',
            'features_arg':[\
             # {'feat_key':'newFeatureName','weight':4,'norm':'zscore_all'},
             {'feat_key':'left_ear','weight':1,'norm':'zscore_all'},
             {'feat_key':'right_ear','weight':1,'norm':'zscore_all'},
             {'feat_key':'left_ear_phi','weight':1,'norm':'zscore_all'},
             {'feat_key':'right_ear_phi','weight':1,'norm':'zscore_all'},
             {'feat_key':'body_length','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_length','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_body_angles','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_rho','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_phi_c','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_phi_s','weight':1,'norm':'zscore_all'},
             {'feat_key':'nose_fft_amp','weight':1,'norm':'zscore_all'},
             {'feat_key':'nose_fft_ang','weight':1,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_amp','weight':1,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_ang','weight':1,'norm':'zscore_all'},
             {'feat_key':'body_change_ang','weight':1,'norm':'zscore_all'},

             {'feat_key':'left_ear_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'right_ear_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'left_ear_phi_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'right_ear_phi_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'body_length_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'head_length_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'head_body_angles_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'displace_rho_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'displace_phi_c_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'displace_phi_s_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'nose_fft_amp_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'nose_fft_ang_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_amp_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_ang_TO','weight':0,'norm':'zscore_all'},
             {'feat_key':'body_change_ang_TO','weight':0,'norm':'zscore_all'},

             {'feat_key':'TM_nose_RM_tail_displace_phi','weight':0,'norm':'zscore_all'},
             {'feat_key':'TM_nose_RM_tail_displace_rho','weight':0,'norm':'zscore_all'},
             {'feat_key':'RM_nose_TM_tail_displace_phi','weight':0,'norm':'zscore_all'},
             {'feat_key':'RM_nose_TM_tail_displace_rho','weight':0,'norm':'zscore_all'},
             {'feat_key':'nose_nose_displace_rho','weight':0,'norm':'zscore_all'},
             {'feat_key':'nose_nose_displace_phi','weight':0,'norm':'zscore_all'},
             {'feat_key':'two_body_ang','weight':0,'norm':'zscore_all'},
             {'feat_key':'two_head_ang','weight':0,'norm':'zscore_all'},
             ],
          },
        ]
```

- Change the threshold for distance between two mice to a very large number

```
self.distance_threshold = 10000 # 
```

- Specify the directory to save the generated videos

```
self.gen_video_folder = # where the generated videos will be saved
```

- Specify the directory to save the metadata which will be used for inspecting the clustering results

```
self.result_folder = # where the metadata will be saved
```

- Specify the suffix for generated videos

```
self.video_name_suffix = # suffix for generated videos
```


#### Configurations for social behavior

- Adjust the weights for Social behavioral features

  ```
  self.cluster_arg = [
        {
            'thred':30, # threshold for defining cluster in the dendrogram
            'name':'all_twoMice',## name: for display
            'evaluation_metric':'Adjusted Rand index',
            'features_arg':[\
             # {'feat_key':'newFeatureName','weight':4,'norm':'zscore_all'},
             {'feat_key':'left_ear','weight':1,'norm':'zscore_all'},
             {'feat_key':'right_ear','weight':1,'norm':'zscore_all'},
             {'feat_key':'left_ear_phi','weight':1,'norm':'zscore_all'},
             {'feat_key':'right_ear_phi','weight':1,'norm':'zscore_all'},
             {'feat_key':'body_length','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_length','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_body_angles','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_rho','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_phi_c','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_phi_s','weight':1,'norm':'zscore_all'},
             {'feat_key':'nose_fft_amp','weight':1,'norm':'zscore_all'},
             {'feat_key':'nose_fft_ang','weight':1,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_amp','weight':1,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_ang','weight':1,'norm':'zscore_all'},
             {'feat_key':'body_change_ang','weight':1,'norm':'zscore_all'},

             {'feat_key':'left_ear_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'right_ear_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'left_ear_phi_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'right_ear_phi_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'body_length_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_length_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_body_angles_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_rho_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_phi_c_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_phi_s_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'nose_fft_amp_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'nose_fft_ang_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_amp_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'contourPCA_fft_ang_TO','weight':1,'norm':'zscore_all'},
             {'feat_key':'body_change_ang_TO','weight':1,'norm':'zscore_all'},

             {'feat_key':'TM_nose_RM_tail_displace_phi','weight':4,'norm':'zscore_all'},
             {'feat_key':'TM_nose_RM_tail_displace_rho','weight':4,'norm':'zscore_all'},
             {'feat_key':'RM_nose_TM_tail_displace_phi','weight':4,'norm':'zscore_all'},
             {'feat_key':'RM_nose_TM_tail_displace_rho','weight':4,'norm':'zscore_all'},
             {'feat_key':'nose_nose_displace_rho','weight':4,'norm':'zscore_all'},
             {'feat_key':'nose_nose_displace_phi','weight':4,'norm':'zscore_all'},
             {'feat_key':'two_body_ang','weight':4,'norm':'zscore_all'},
             {'feat_key':'two_head_ang','weight':4,'norm':'zscore_all'},
             ],
          },
        ]
  ```

- Change the threshold for distance between two mice to include only mice that are within 1.8 body length

```
self.distance_threshold = 1.8
```


- Specify the directory to save the generated videos

```
self.gen_video_folder = # where the generated videos will be saved
```

- Specify the directory to save the metadata which will be used for inspecting the clustering results

```
self.result_folder = # where the metadata will be saved
```

- Specify the suffix for generated videos

```
self.video_name_suffix = # suffix for generated videos
```
 
 <br>


### step 2. Preprocess Data

Run the following command in the current folder

```bash
cd utils_file
bash run_all.sh
```

### step 3. Clustering

Run the following command in the current folder

```bash
python fft_main_sep_twoMiceInteract.py
```

<br>


## Add new features (optional)

There are two steps to add feature for clustering:

###  Step 1. Add Code for the Feature

1. Define the new feature ./fft_utils.py as the following template (you can find this template in ./fft_utils.py ；

```
        ## you can process data to get the feature
        ## the pose data are in five variables: pose_clips, pose_clips_align, poseTheOther_clips, poseTheOther_clips_alignSelf, poseTheOther_clips_alignToOther
        ## each of the variables is a numpy array whose shape is (number_of_clip, number_of_frames_in_one_clip, number_of_key_point, 3)
        ## pose_clips contains the raw key points data of the target mouse
        ## pose_clips_align contains key point data that is aligned to the target mouse in the middle frame
        ## poseTheOther_clips contains the raw key points data of the mouse that is closest to the target mouse
        ## poseTheOther_clips_alignSelf contains the key points data of the closest mousethat that is aligned to itself in the middle frame
        ## poseTheOther_clips_alignToOther contains the key points data of the closest mousethat that is aligned to the target mouse in the middle frame
        newFeature = np.ones(pose_clips.shape[1])  # the feature of one clip should be a numpy array whose shape is the frame number of the clip
        if 'newFeatureName' in feature_clips_dict:
            feature_clips_dict['newFeatureName'].append(newFeature)
        else:
            feature_clips_dict['newFeatureName'] = [newFeature]
```

2. define the weight and normalization of the new feature in ./utils_file/setting.py :
```python
	self.cluster_arg = [
           {
            'thred':30,
            'name':'all_twoMice',
            'evaluation_metric':'Adjusted Rand index',
            'features_arg':[\
             {'feat_key':'newFeatureName','weight':4,'norm':'zscore_all'}, # add the setting of the new feature here

             {'feat_key':'body_length','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_length','weight':1,'norm':'zscore_all'},
             {'feat_key':'head_body_angles','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_rho','weight':1,'norm':'zscore_all'},
             {'feat_key':'displace_phi_c','weight':1,'norm':'zscore_all'},
             ],
             },
        ]
```
<br>

### step 2. Clustering

Run the following command in the current folder
```bash
python fft_main_sep_twoMiceInteract.py
```

<br>


## Inspect Clustering results

### Install the following packages with conda

```
- pprint
- json
- pickle
- numpy
- pandas
- matplotlib
- scipy
```

<br>

### Run Analysis.ipynb to generate the following plots

Detailed instructions are included in the jupyter notebook

- Dendrogram

- Timeline

- UMAP

- Mutual information plots

- Similarity matrix between clusters

<br>

### Run Visualize.ipynb to visualize skeletons for each cluster

Detailed instructions are included in the jupyter notebook

- Individual cluster skeletons

- Social behaviror cluster skeletons
