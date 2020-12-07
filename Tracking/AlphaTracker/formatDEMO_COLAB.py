import os
import json
import shutil
from datetime import datetime

from google_drive_downloader import GoogleDriveDownloader as gdd
sppe_id = '1_BwtYySpX9uWDgdwqw0UEppyMYYv1gkJ'
yolo_id = '13zXkuZ4dNm3ZOwstr1sSWKOOzJ19XZpN'


def make_settingPY():
    
	if os.path.exists('/content/drive/My Drive/data.zip'):
	os.remove('/content/drive/My Drive/data.zip')

	if os.path.exists('/content/drive/My Drive/demo_video.mp4'):
	os.remove('/content/drive/My Drive/demo_video.mp4')

	if os.path.exists('/content/drive/My Drive/TRAINING_DATA'):
	shutil.rmtree('/content/drive/My Drive/TRAINING_DATA')

	data_zip_link = '15dR-vVCEsg2z7mEVzJOF9YDW6YioEU3N'
	data_video_link = '1N0JjazqW6JmBheLrn6RoDTSRXSPp1t4K'
	sppe_id = '1_BwtYySpX9uWDgdwqw0UEppyMYYv1gkJ'
	yolo_id = '13zXkuZ4dNm3ZOwstr1sSWKOOzJ19XZpN'

	gdd.download_file_from_google_drive(file_id=data_zip_link, dest_path='/content/drive/My Drive/data.zip')
	gdd.download_file_from_google_drive(file_id=data_video_link, dest_path='/content/drive/My Drive/demo_video.mp4')

	gdd.download_file_from_google_drive(file_id=sppe_id, dest_path='/gdrive/AlphaTracker/Tracking/AlphaTracker/model10.pkl')

	gdd.download_file_from_google_drive(file_id=yolo_id, dest_path='/gdrive/AlphaTracker/Tracking/AlphaTracker/yolov3-mice_final.weights')

	#new_video_paths = [s.replace('/content/drive/My Drive', '/gdrive') for s in video_path];
	
	pose_pair = [[0,1],[0,2],[0,3]]
	train_val_split = 0.90
	with open('setting.py', 'w') as f:
		f.write("import os\n")
		f.write("gpu_id=0\n")
		f.write("AlphaTracker_root = '/gdrive/AlphaTracker/Tracking/AlphaTracker'\n")
		f.write("image_root_list=['/gdrive/TRAINING_DATA/demo']\n")
		f.write("json_file_list=['/gdrive/TRAINING_DATA/demo/train9.json']\n")
		f.write("num_mouse=[2]\n")
		f.write("num_pose=4\n")
		f.write("exp_name='DEMO'\n")
		f.write("pose_pair={}\n".format(pose_pair))
		f.write("train_val_split=\n".format(train_val_split))
		f.write("image_suffix='jpg'\n")
		f.write("sppe_lr=1e-4\n")
		f.write("sppe_epoch=5\n")
		f.write("sppe_batchSize=10\n")
		f.write("sppe_pretrain='/gdrive/AlphaTracker/Tracking/AlphaTracker/model10.pkl'\n")
		f.write("yolo_lr=0.0005\n")
		f.write("yolo_iter=100\n")
		f.write("yolo_batchSize=4\n")
		f.write("yolo_pretrain='/gdrive/AlphaTracker/Tracking/AlphaTracker/yolov3-mice_final.weights'\n")

		f.write("video_full_path='/gdrive/demo_video.mp4'\n")

		f.write("start_frame=0\n".format(start_frame))
		f.write("end_frame=1000\n")
		f.write("max_pid_id_setting=2\n")
		f.write("result_folder='/gdrive/result_folder'\n")
		f.write("remove_oriFrame=False\n")

		f.write("vis_track_result=1\n")
		f.write("weights = '0 6 0 0 0 0 '\n")
		f.write("match = 0\n")
