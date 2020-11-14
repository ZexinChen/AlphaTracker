# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt


class Mscoco(data.Dataset):
    def __init__(self,img_folder= '../data/coco/images/', annot_file='../data/coco/annot_coco.h5', 
                 train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian',nJoints=8):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = nJoints
        self.nJoints = nJoints

        self.accIdxs = (1, 2, 3, 4)
        self.flipRef = [(2, 3)]

        # create train/val split
        with h5py.File(annot_file, 'r') as annot:
            if self.is_train:
                # train
                self.imgname_coco_train = annot['imgname'][:]
                self.bndbox_coco_train = annot['bndbox'][:]
                self.part_coco_train = annot['part'][:]
                self.size_train = self.imgname_coco_train.shape[0]
            else:
                # val
                self.imgname_coco_val = annot['imgname'][:]
                self.bndbox_coco_val = annot['bndbox'][:]
                self.part_coco_val = annot['part'][:]
                self.size_val = self.imgname_coco_val.shape[0]

            # print(self.imgname_coco_train[1])
            # print(reduce(lambda x, y: x + y,
            #              map(lambda x: chr(int(x)), annot['imgname'][0])))


    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            # print('self.part_coco_train[index]:::',self.bndbox_coco_train)
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]

        # print(self.imgname_coco_train[4])
        # print(imgname)
        imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
        img_path = os.path.join(self.img_folder, imgname)
        # print(img_path,imgname,index)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'coco', sf, self, train=self.is_train,nJoints_coco=self.nJoints_coco)

        inp, out, setMask = metaData

        return inp, out, setMask, 'coco'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
