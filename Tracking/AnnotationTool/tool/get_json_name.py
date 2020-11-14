import os
import re
import json
import numpy as n


im_dir = './im/'
json_label_dir = './'

im_list = n.sort(os.listdir(im_dir))
json_label_path = os.path.join(json_label_dir, 'multi_person.json')
fs = open(json_label_path, 'w')
annotations= []
for im in im_list:

    annotations.append({'annotations': [], 'class': 'image', 'filename': im})
json.dump(annotations, fs, indent=4, separators=(',', ': '), sort_keys=True)
fs.write('\n')

fs.close()
print 'Done!'