import os
import sys
import cv2

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if os.path.exists(path) and os.path.isdir(path):
            pass
        else: raise

# video_full_path_list = [
#     '/disk4/zexin/project/mice/datasets/0520/2036_black_four.mov',
#     '/disk4/zexin/project/mice/datasets/0519/2026_black_four.mov',
#     '/disk4/zexin/project/mice/datasets/0521/1953_black_four_male.mov'
#     ]
# frame_esp_list =[
#     [
#     [1300-30,   1500+30],
#     [2200-30,   2350+30],
#     [3550-30,   3800+30],
#     [6600-30,   6680+30],
#     [7400-30,   7680+30],
#     [8700-30,   9070+30],
#     [10400-30, 10500+30],
#     [10680-30, 11080+30],
#     [18600-30, 18700+30]
#     ],
#     [
#     [11840-30, 12130+30],
#     [21720-30, 22000+30]
#     ],
#     [
#     [1030-30,   1090+30],
#     [1830-30,   1920+30],
#     [1990-30,   2050+30],
#     [4900-30,   5010+30],
#     [7010-30,   7070+30],
#     [7300-30,   7390+30],
#     [9150-30,   9270+30]
#     ]

# ]
# save_root = '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/select_track_frame/'

# video_full_path_list = [
#     '/disk4/zexin/project/mice/datasets/0525/1915_black_five.mov',
#     '/disk4/zexin/project/mice/datasets/0525/1854_black_five.mov',
#     ]
# frame_esp_list =[
#     [
#     [0  ,390+30],
#     [950-30,1100+30],
#     [1830-30,  2210+30],
#     [2700-30,  3280+30],
#     [3490-30,  3580+30],
#     [3740-30,  4000+30],
#     [4160-30,  4700+30]
#     ],
#     [
#     [160-30,    240+30],
#     [700-30,    900+30],
#     [1150-30,   1290+30],
#     [1300-30,   1510+30],
#     [1870-30,   2070+30],
#     [2200-30,   2320+30],
#     [2840-30,   3040+30],
#     [3980-30,   4260+30],
#     [4520-30,   5100+30],
#     [5800-30,   5920+30],
#     [7450-30,   7720+30],
#     [8045-30,   8078+30],
#     [8500-30,   8610+30],
#     [9510-30,   9600+30],
#     [9700-30,   9860+30],
#     [11111-30,  11390+30],
#     [11620-30,  11710+30],
#     [11800-30,  11910+30],
#     [12000-30,  12270+30],
#     [13428-30,  13747+30],
#     [14211-30,  14500+30]
#     ]
# ]
# save_root = '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/select_track_frame/'
# frame_distance = 6


video_full_path_list = [
'/disk4/zexin/project/mice/datasets/06_040506_twoMice/0902_black_two.mov',
'/disk4/zexin/project/mice/datasets/06_040506_twoMice/0854_black_two.mov',
'/disk4/zexin/project/mice/datasets/06_040506_twoMice/0910_black_two.mov',
]
frame_esp_list =[
    [
        [6300,6500],
        [6900,7500],
        [7900,8700],
        [9380,9850],
        [10500,11600],
        [11800,12600],
        [13600,13700],
        [14200,14400],
    ],
    [
        [990,1800],
        [2800,3800],
        [3950,4100],
        [4200,4300],
        [4600,4900],
        [5300,5600],
    ],
    [
        [2900,2980],
    ]
]
save_root = '/disk4/zexin/datasets/mice/new_labeled_byCompany/08/select_track_frame/'
frame_distance = 6

al_count = 0
for a in frame_esp_list:
    for b in a:
        al_count = al_count + (b[1]-b[0])

print('total frame:',al_count/frame_distance)

for video_i in range(len(video_full_path_list)):
    video_full_path = video_full_path_list[video_i]
    print('reading video:',video_full_path)
    cap = cv2.VideoCapture(video_full_path)
    if cap.isOpened():
        success = True
    else:
        success = False
        print(" read failed!make sure that the video format is supported by cv2.VideoCapture")
        exit()

    frame_index = -1
    for fe_i in range(len(frame_esp_list[video_i])):
        one_frame_esp = frame_esp_list[video_i][fe_i]
        video_image_save_path = save_root+'/'+video_full_path.split('/')[-2]+'_'+video_full_path.split('/')[-1]+'_'+str(one_frame_esp[0])+'_'+str(one_frame_esp[1])
        if not os.path.exists(video_image_save_path):
            mkdir_p(video_image_save_path)
        else:
            # os.system('rm {}/*'.format(video_image_save_path))
            pass
        
        while (success):
            success, frame = cap.read()

            if not success:
                print('read frame failed!')
                break
            frame_index += 1
            sys.stdout.write('\r>> Converting image %d' % (
                frame_index,))
            sys.stdout.flush()

            if frame_index<one_frame_esp[0]:
                continue
            elif frame_index>=one_frame_esp[1]:
                break
            elif (frame_index-one_frame_esp[0])%frame_distance == 0:
                frame_path = video_image_save_path+'/'+video_full_path.split('/')[-2]+'_'+video_full_path.split('/')[-1].split('.')[0]+'_'+video_full_path.split('/')[-1].split('.')[1]+'_'+str(one_frame_esp[0])+'_'+str(one_frame_esp[1])+'_'+str(frame_index)+'.png'
                cv2.imwrite(frame_path, frame)
                print('\r saving frame as ',frame_path,end='')
            else:
                continue

    cap.release()
print('')
