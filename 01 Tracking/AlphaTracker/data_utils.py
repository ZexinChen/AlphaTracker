import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import time

import json
import h5py
from tqdm import tqdm


def merge_clean_ln_split_Data(image_root_list,json_file_list,ln_image_dir,train_val_split,num_mouse,num_pose,valid_image_root):
    num_badAnnot = 0
    num_allAnnot_train = 0
    num_allAnnot_valid = 0
    assert len(image_root_list)==len(json_file_list),'len of image_root_list and len of json_file_list should be equal'
    train_data = []
    valid_data = []
    for ii in range(len(json_file_list)):
        single_img_data_count = 0
        json_file = json_file_list[ii]
        print('loading data from ',json_file)
        with open(json_file, 'r') as File:
            data = json.load(File)

        ### clean data
        new_data = []
        for single_img_data in tqdm(data):
            try:
                name=single_img_data['filename']
                annot=single_img_data['annotations']

                new_annot = []
                has_box = False
                for idx in range(len(annot)):
                    if annot[idx]['class']=='Face' or annot[idx]['class']=='boundingBox' or annot[idx]['class']=='point':
                        new_annot.append(annot[idx]) 
                    if annot[idx]['class']=='Face' or annot[idx]['class']=='boundingBox':
                        has_box = True
                annot = new_annot 
                if(len(new_annot)!=num_mouse[ii]*(num_pose+1)):
                    num_badAnnot += 1
                    continue
                if(not has_box):
                    num_badAnnot += 1
                    continue
                for mice_id in range(num_mouse[ii]):
                    d=annot[mice_id*(num_pose+1)]
                    bbox = [d['x'], d['y'], d['x']+d['width'], d['y']+d['height']][:]
                    pt = [[annot[mice_id*(num_pose+1)+k+1]['x'], annot[mice_id*(num_pose+1)+k+1]['y']] for k in range(num_pose)][:]
                    iname = [ord(x) for x in name][:]
                ## checked
                os.system('ln -s {}/{} {}/{}'.format(\
                                 image_root_list[ii],single_img_data['filename'],\
                                 ln_image_dir,'%04d'%(ii)+'_'+'%06d'%(single_img_data_count)+single_img_data['filename'][-4:]))
                single_img_data['filename'] = '%04d'%(ii)+'_'+'%06d'%(single_img_data_count)+single_img_data['filename'][-4:]
                new_data.append(single_img_data)
                single_img_data_count += 1
                # num_allAnnot += num_mouse[ii]
            except:
                num_badAnnot += 1
                print('bad annot!!:',single_img_data)
        
        ### split data
        train_data += new_data[:int(len(data)*train_val_split)]
        valid_data += new_data[int(len(data)*train_val_split):]
        num_allAnnot_train += len(new_data[:int(len(data)*train_val_split)])*num_mouse[ii]
        num_allAnnot_valid += len(new_data[int(len(data)*train_val_split):])*num_mouse[ii]

        ###ln valid image to valid_image_root
        for vd in new_data[int(len(data)*train_val_split):]:
            os.system('ln -s {}/{} {}/{}'.format(\
                                 ln_image_dir,vd['filename'],\
                                 valid_image_root,vd['filename']))
                

    print('total_bad annot:',num_badAnnot)
    return train_data,valid_data,num_allAnnot_train,num_allAnnot_valid



def generate_h5(h5_file,in_data,num_allAnnot,num_pose,num_mouse):
    f = h5py.File(h5_file, "w")

    bbox = f.create_dataset("bndbox", (num_allAnnot, 1, 4,), dtype='i')
    iname = f.create_dataset("imgname", (num_allAnnot,15, ), dtype='i')
    pt = f.create_dataset("part", (num_allAnnot, num_pose, 2,), dtype = 'i')

    # # Creating .h5 file
    ind=0
    for single_img_data in tqdm(in_data):
        name=single_img_data['filename']
        annot=single_img_data['annotations']

        ann_idx = 0
        new_annot = []
        for idx in range(len(annot)):
            if annot[idx]['class']=='Face' or annot[idx]['class']=='boundingBox' or annot[idx]['class']=='point':
                new_annot.append(annot[idx])


        annot = new_annot 
        # print(int(len(annot)/(num_pose+1)))
        for mice_id in range(int(len(annot)/(num_pose+1))):
            d=annot[mice_id*(num_pose+1)]
            try:
                bbox[ind] = [d['x'], d['y'], d['x']+d['width'], d['y']+d['height']][:]
            except Exception as e:
                print(e)
                print('error!,bbox format error!',d,bbox)

            pt[ind] = [[annot[mice_id*(num_pose+1)+k+1]['x'], annot[mice_id*(num_pose+1)+k+1]['y']] for k in range(num_pose)][:]
            
            # print([ord(x) for x in name])
            iname[ind] = [ord(x) for x in name][:]
            ind=ind+1
    print(ind)


    f.close()
    print('writing data to ',h5_file)


def generate_yolo_data(list_file,data_in,image_root_in,yolo_annot_root,image_suffix,color_img_prefix):
    with open(list_file,'w') as traintxt:
        for item in tqdm(data_in):
            # print(image_root_in+item['filename'])
            img=cv2.imread(image_root_in+item['filename'])
            try:
                yolo_inputImg_size = (img.shape[1],img.shape[0])  
            except Exception as e:
                print('image path:')
                print(image_root_in+item['filename'])
                raise e
            with open((yolo_annot_root+item['filename'].strip(image_suffix)+'txt'),'w') as fileout:
                for j in item['annotations']:
                    if (j['class']=='Face') or j['class']=='boundingBox':
                        x_mean=(j['x']+j['width']/2)/yolo_inputImg_size[0]
                        y_mean=(j['y']+j['height']/2)/yolo_inputImg_size[1]
                        width=j['width']/yolo_inputImg_size[0]
                        height=j['height']/yolo_inputImg_size[1]
                        content='0'+' '+str(x_mean)+' '+str(y_mean)+' '+str(width)+' '+str(height)+'\n'
                        fileout.writelines(content)
            traintxt.writelines(color_img_prefix+item['filename']+'\n')


def generate_evalJson(in_data,image_root_in,num_pose):
    gt_dict = {}
    gt_dict['annotations'] = []
    gt_dict['images'] = []
    gt_dict["categories"] = [{
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": ["nose","left_ear","right_ear","tail"]}]
    gt_dict['skeleton']= [[1,2]]
    gt_dict['num_keypoints']=num_pose
    ind=0
    count = 0
    pred_dict_makeup = []
    all_image_count = 0
    all_annot_count = 0
    for single_img_data in tqdm(in_data):
        name=single_img_data['filename']
        annot=single_img_data['annotations']

        img = cv2.imread(image_root_in+'/'+name)

        new_annot = []
        for idx in range(len(annot)):
            if annot[idx]['class']=='Face' or annot[idx]['class']=='boundingBox' or annot[idx]['class']=='point':
                new_annot.append(annot[idx])      
        
        annot = new_annot

        all_image_count += 1
        one_image_dict = {
            "file_name":name,\
            'width':img.shape[0], \
            "height":img.shape[1], \
            "id":all_image_count\
        }
        gt_dict['images'].append(one_image_dict)

        for mice_id in range(int(len(annot)/(num_pose+1))):
            keypoints = []
            for k in range(4):
                keypoints = keypoints + [int(annot[mice_id*5+k+1]['x']), int(annot[mice_id*5+k+1]['y']), 2]
            d=annot[mice_id*5]
            bbox = [int(d['x']), int(d['y']), int(d['width']), int(d['height'])]

            all_annot_count += 1
            one_gt_dict = {
                "image_id": all_image_count, \
                "category_id": 1, \
                "bbox": bbox, \
                "area": d['width']*d['height']*0.8,
                "keypoints": keypoints,\
                "num_keypoints":4, \
                "iscrowd":0, \
                "score": 1,\
                "id":all_annot_count,\
                "file_name":name
            }
            gt_dict['annotations'].append(one_gt_dict)
    return gt_dict
    








