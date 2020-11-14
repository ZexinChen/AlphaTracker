import os
import json
import shutil
from datetime import datetime

def convert(image_filepaths, json_filepaths, extension):
    save_path = '/content/drive/My Drive/TRAINING_DATA'+'_'+datetime.today().strftime('%Y-%m-%d')
    if os.path.isdir(save_path) == True:
        print('Directory already exists...please adjust before continuing!')
    else:
        os.mkdir(save_path)
    
    comb_pop = []
    for j in range(0, len(image_filepaths)):
    #j = 0
        with open(json_filepaths[j]) as f:
            data = json.load(f)
    
        pop = []
        single_img_data_count = 0
        for i in data:
            img_name = i['filename']
            full_path = os.path.join(image_filepaths[j], img_name)
            new_name = '%04d'%(j)+'_'+'%06d'%(single_img_data_count)+'.'+extension
    
            i['filename'] = new_name
            
            if os.path.isfile(full_path):
                shutil.copy(full_path, save_path)
                
            reloc_path = os.path.join(save_path, img_name)
            os.rename(reloc_path, os.path.join(save_path, new_name))
            #os.rename(full_path, os.path.join(image_filepaths[j], new_name))
    
            single_img_data_count += 1
            pop.append(i)
            
        comb_pop.append(pop)
        
    comb_pop = [x for y in comb_pop for x in y]
    
    json_save = save_path + '/ATjsonCOLAB.json'
    with open(json_save, 'w') as outfile:
        json.dump(comb_pop, outfile, indent=4)