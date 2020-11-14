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

import contour_utils
import setting

def create_dataframe_faces():

    # 取读所有的mask图，构成一个dataframe


    faces = pd.DataFrame([])
    i = 0
    arg = setting.args_class()

    for video_path,json_path,contour_path in zip(arg.videodir,arg.tracked_json,arg.contdir):
        img_dict = contour_path
        data = contour_utils.load_json(json_path)

        for img_name in tqdm(os.listdir(img_dict)):

            if not(img_name[-1] == 'g'):
                print(img_name+' is not used')
                continue

            if np.random.randn()>0.8:
                print('add %d th image '%(len(faces+1)))
                # print(i,img_dict + img_name)
                frame = plt.imread(img_dict + img_name)
                frame = np.asarray(frame[:,:,0],dtype = 'uint8')
                frame[frame == 255] = 1
                # flat = frame.flatten()

                # print(flat.shape)
                face = pd.Series(frame.flatten(),name = img_name)

                # print(face)
                faces = faces.append(face)
                i += 1
            if len(faces)>2500:
                break
        if len(faces)>2500:
            break


    # ## 可视化
    # width, height = frame.shape
    # fig, axes = plt.subplots(10,10,figsize=(9,9),
    #     subplot_kw = {'xticks':[], 'yticks':[]},
    #     gridspec_kw = dict(hspace=0.01, wspace=0.01))

    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(faces.iloc[i].values.reshape(height,width),cmap='gray')

    # plt.savefig('love.png')

    import pickle
    print('writing file...')
    with open('./faces.pckl', 'wb') as f:
        pickle.dump(faces, f)


if __name__ == '__main__':
    create_dataframe_faces()



