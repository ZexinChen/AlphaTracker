import os
import cv2 
import numpy as np
from tqdm import tqdm

# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/webcam_505_video1/'
# video_full_path = '/disk4/zexin/project/mice/datasets/webcam_505.mov'
# save_image_format = "%d.png"
# start_frame = 800
# end_frame = 1600

# video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1428.mov'
# start_frame = 1600
# end_frame = 2400
# image_folder_name = '0507_webcam_1428_mov_%d_%d'%(start_frame,end_frame)
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = "%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1624.mov'
# start_frame = 0
# end_frame = 2400
# interval = 5
# image_folder_name = '0507_webcam_1624_mov_%d_%d'%(start_frame,end_frame)
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1504.mov'
start_frame = 0
end_frame = 2400
interval = 5
image_folder_name = '0507_webcam_1504_mov_%d_%d'%(start_frame,end_frame)
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1504.mov'
start_frame = 3000
end_frame = 5500
interval = 5
image_folder_name = '0507_webcam_1504_mov_%d_%d'%(start_frame,end_frame)
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1551.mov'
start_frame = 0
end_frame = 2500
interval = 5
image_folder_name = '0507_webcam_1551_mov_%d_%d'%(start_frame,end_frame)
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1347.mov'
start_frame = 0
end_frame = 2500
interval = 5
image_folder_name = '0507_webcam_1347_mov_%d_%d'%(start_frame,end_frame)
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1227.mov'
video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1227.mov'
start_frame = 0
end_frame = 15000
interval = 30
image_folder_name = '0507_webcam_1227_mov_%d_%d'%(start_frame,end_frame)
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"
sample_rate = 0.1


video_full_path = '/disk4/zexin/project/mice/datasets/0507_webcam/1347.mov'
start_frame = 0
end_frame = 15000
interval = 30
image_folder_name = '0507_webcam_1347_mov_%d_%d'%(start_frame,end_frame)
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"
sample_rate = 0.1


if not os.path.exists(video_image_save_path):
    os.mkdir(video_image_save_path)
else:
    os.system('rm {}/*'.format(video_image_save_path))
cap = cv2.VideoCapture(video_full_path)
frame_index = 0
frame_count = 0
if cap.isOpened():
    success = True
else:
    success = False
    print("read video failed!make sure that the video format is supported by cv2.VideoCapture")

# # while(success):
# save_count = 0
# last = start_frame
# save_frame_idxs = [ii+np.random.randint(0,interval-1) for ii in range(start_frame,end_frame,interval)]
# random_idx = np.random.randint(0,interval-1)
# for frame_index in tqdm(range(end_frame)):
#     success, frame = cap.read()
# #     if frame_index % interval == random_idx:
#     if frame_index < start_frame:
#         continue
#     if frame_index in save_frame_idxs:
#     # if frame_index > last+ interval and np.random.random_sample()<0.01:
#         # last = frame_index
# #         visImage(frame)
# #         resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
# #         # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
# #         cv2.imwrite(video_image_save_path + "%d.jpg" % frame_count, frame)
        
#         cv2.imwrite(video_image_save_path +  save_image_format % frame_index, frame)
#         random_idx = np.random.randint(0,interval-1)
#         save_count+=1
#         if not success:
#             print('read frame failed!')
#             break

save_count = 0
last = start_frame
random_idx = np.random.randint(0,interval-1)
frame_index = -1
while(success):
    success, frame = cap.read()
    frame_index += 1
    if frame_index<start_frame:
        continue
    if frame_index>end_frame:
        break
    if frame_index > last+ interval and np.random.random_sample()<sample_rate:
        last = frame_index
        
        cv2.imwrite(video_image_save_path +  save_image_format % frame_index, frame)
        random_idx = np.random.randint(0,interval-1)
        save_count+=1
        print('saving the %d frame (%d)'%(frame_index, save_count))


        if not success:
            print('read %d frame failed!'%frame_index)

cap.release()
print('save_count:',save_count)



