import os
import cv2

from tqdm import tqdm

video_full_path = './1411_black_two.mov.mov'
video_image_save_path = './1411_black_two_frame/'
save_image_format = "frame_%d.png"


print('processing %s'%(video_full_path))    
cap = cv2.VideoCapture(video_full_path)
if cap.isOpened():
    success = True
else:
    success = False
    print(" read failed!make sure that the video format is supported by cv2.VideoCapture")

# while(success):
for frame_index in tqdm(range(1000)):
    success, frame = cap.read()
    if not success:
        print('read frame failed!')
        break
    cv2.imwrite(video_image_save_path +  save_image_format % frame_index, frame)
    
cap.release()

