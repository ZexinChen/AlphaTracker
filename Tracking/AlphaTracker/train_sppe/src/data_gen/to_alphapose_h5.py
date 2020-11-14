import json
import h5py
with open('annotation.json', 'r') as File:
    data = json.load(File)

f = h5py.File("data.h5", "w")

valid_len=len(data)-2# incorrect annotation will be neglected

bbox = f.create_dataset("bndbox", (valid_len*2, 1, 4,), dtype='i')
iname = f.create_dataset("imgname", (valid_len*2, 8, ), dtype='i')
#note the original version on github  has size 16, langwen used 8. Actual file names have length 8.

pt = f.create_dataset("part", (valid_len*2, 4, 2,), dtype = 'i')

# Creating .h5 file
ind=0

for single_img_data in data:
    name=single_img_data['filename']
    annot=single_img_data['annotations']

    if(len(annot)!=10):
        print("wtf!"+name)
        print("this annotation have incorect amount of keypoints")
        continue

    for i in range(len(annot)):
        if(annot[i]['class']=='Face'):
            d=annot[i]
            bbox[ind] = [d['x'], d['y'], d['x']+d['width'], d['y']+d['height']][:]
            iname[ind] = [ord(x) for x in name][:]
            pt[ind] = [[annot[i+k+1]['x'], annot[i+k+1]['y']] for k in range(4)][:]
            #folowing four entries are keypoints for that mouse
            ind=ind+1
        