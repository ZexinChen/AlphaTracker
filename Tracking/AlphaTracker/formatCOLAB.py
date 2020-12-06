import os
import json
import shutil
from datetime import datetime

def convert(image_filepaths, json_filepaths, extension):
    save_path = '/content/drive/My Drive/TRAINING_DATA'
    if os.path.isdir(save_path) == True:
        shutil.rmtree(save_path)
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
		
		
		
def make_settingPY(image_root_list, json_file_path, num_mouse, num_pose, exp_name, image_suffix, sppe_lr, \
                   sppe_epoch, sppe_batchSize, yolo_lr, yolo_iter, yolo_batchSize, start_frame, end_frame, \
				   max_pid_setting, visualize, video_path):

				   
	new_video_paths = [s.replace('/content/drive/My Drive', '/gdrive') for s in video_path];
	
	pose_pair = [[0,1],[0,2],[0,3]]
	train_val_split = 0.90
	with open('setting.py', 'w') as f:
		f.write("import os\n")
		f.write("gpu_id=0\n")
		f.write("AlphaTracker_root = '/gdrive/AlphaTracker/Tracking/AlphaTracker'\n")
		f.write("image_root_list=['/gdrive/TRAINING_DATA']\n")
		f.write("json_file_list=['/gdrive/TRAINING_DATA/ATjsonCOLAB.json']\n")
		f.write("num_mouse={}\n".format(num_mouse))
		f.write("num_pose={}\n".format(num_pose))
		f.write("exp_name='{}'\n".format(exp_name))
		f.write("pose_pair={}\n".format(pose_pair))
		f.write("train_val_split={}\n".format(train_val_split))
		f.write("image_suffix='{}'\n".format(image_suffix))
		f.write("sppe_lr={}\n".format(sppe_lr))
		f.write("sppe_epoch={}\n".format(sppe_epoch))
		f.write("sppe_batchSize={}\n".format(sppe_batchSize))
		f.write("sppe_pretrain=''\n")
		f.write("yolo_lr={}\n".format(yolo_lr))
		f.write("yolo_iter={}\n".format(yolo_iter))
		f.write("yolo_batchSize={}\n".format(yolo_batchSize))
		f.write("yolo_pretrain=''\n")
		
		if len(video_path) == 1:
			f.write("video_full_path='{}'\n".format(new_video_paths[0]))
		else:
			f.write("video_paths={}\n".format(new_video_paths))
			
		f.write("start_frame={}\n".format(start_frame))
		f.write("end_frame={}\n".format(end_frame))
		f.write("max_pid_id_setting={}\n".format(max_pid_setting))
		f.write("result_folder='/gdrive/result_folder'\n")
		f.write("remove_oriFrame=False\n")
		
		if visualize == 'yes':
			f.write("vis_track_result=1\n")
		else:
			f.write("vis_track_result=0\n")
			
		f.write("weights = '0 6 0 0 0 0 '\n")
		f.write("match = 0\n")
