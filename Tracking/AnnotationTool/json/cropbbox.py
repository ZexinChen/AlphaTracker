__author__ = 'develop'

import os
import json
from PIL import Image
from math import floor

ann_dir = './im/multi_person.json'
out_dir = './crop_test'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
im_dir = './im'
fjson = open(ann_dir, 'r')
ann = json.load(fjson)
im_num = 0
for d in ann:
    im_name = d['filename']
    im_num += 1
    #import pdb
    #pdb.set_trace()
    im_path = os.path.join(im_dir,im_name)
    #save_crop_name = os.path.join(out_dir,im_name,'%d'%)

    #im = cv2.imread(im_path)
    im = Image.open(im_path)
    imgw,imgh = im.size
    now = 0
    if d['annotations'] ==[]:
        continue
    for elem in d['annotations']:
        save_crop_name = os.path.join(out_dir,im_name[:-4]+'_%d'%now+im_name[-4:])
        x1 = max(int(elem['x']),0)
        y1 = max(int(elem['y']),0)
        x2 = min(x1 + int(elem['width']),imgw)
        y2 = min(y1 + int(elem['height']),imgh)
        if int(elem['width'])*int(elem['height']) ==0:
            continue
        #import pdb
        #pdb.set_trace()
        #cv2.rectangle(im,(x1, y1), (x2, y2), (0, 255, 0), 3)
        rate = max((y2 - y1 + 1)/1500.0,1.0)
        r_imgw =  int(floor((x2 - x1 + 1)/rate))
        r_imgh =  int(floor((y2 - y1 + 1)/rate))
        rim = im.crop((x1, y1, x2, y2)).resize((r_imgw,r_imgh),Image.ANTIALIAS)
        #crop_im = im[y1:y2,x1:x2,:]
        #elem[]
        rim.save(save_crop_name)
        #cv2.imwrite(save_crop_name,crop_im)
        now += 1
        #import pdb
        #pdb.set_trace()
    print '%d------> of ------>%d'%(im_num,len(ann))
fjson.close()
