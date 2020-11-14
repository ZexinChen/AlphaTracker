# %matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import time

import json
# import h5py
from tqdm import tqdm


#########################################################################################################
###                                  demo video setting                                               ###
#########################################################################################################

video_full_path = '/disk4/zexin/project/mice/datasets/0519/1807_black_four.mov'
start_frame = 0
end_frame = 3000
image_folder_name = '0519_1807_black_four_mp4_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 4
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0519/1807_black_four.mov'
start_frame = 3000
end_frame = 15000
image_folder_name = '0519_1807_black_four_mp4_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 4
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"


video_full_path = '/disk4/zexin/project/mice/datasets/0519/1728_black_four.mov'
start_frame = 1800
end_frame = 3600
image_folder_name = '0518_1728_black_four_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 4
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"


video_full_path = '/disk4/zexin/project/mice/datasets/0519/2052_black_four.mov'
start_frame = 0
end_frame = 2700
image_folder_name = '0518_2052_black_four_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 4
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"


video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1523_two_black.mov'
start_frame = 0
end_frame = 10000
image_folder_name = '0518_1523_two_black_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1533_two_black.mov'
start_frame = 0
end_frame = 10000
image_folder_name = '0518_1533_two_black_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1353_two_black.mov'
start_frame = 0
end_frame = 600
image_folder_name = '0518_1353_two_black_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0519/2026_black_four.mov'
# start_frame = 11700
# end_frame = 11700+30*20
# image_folder_name = '0518_2026_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0519/2026_black_four.mov'
# start_frame = 21600
# end_frame = 21600+30*20
# image_folder_name = '0518_2026_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0519/2026_black_four.mov'
# start_frame = 0
# end_frame = 20000
# image_folder_name = '0518_2026_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0519/1826_black_four.mov'
# start_frame = 0
# end_frame = 20000
# image_folder_name = '0518_1826_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Four/2135_four_black.mov'
# start_frame = 0
# end_frame = 20000
# image_folder_name = '0518_2135_four_black_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Four/2135_four_black.mov'
# start_frame = 0
# end_frame = 20000
# image_folder_name = '0518_2135_four_black_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Four/2200_four_black.mov'
# start_frame = 0
# end_frame = 20000
# image_folder_name = '0518_2200_four_black_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0510_male/1313.mov'
# start_frame = 10042-30*10
# end_frame = 10042+30*10
# image_folder_name = '0510_male_1313_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0510_male/1027.mov'
# start_frame = 0
# end_frame = 5000
# image_folder_name = '0510_male_1313_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1353_two_black.mov'
# start_frame = 10000
# end_frame = 20000
# image_folder_name = '0518_1353_two_black_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 2
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1533_two_black.mov'
# start_frame = 10000
# end_frame = 20000
# image_folder_name = '0518_1533_two_black_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 2
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2036_black_four.mov'
# start_frame = 1000
# end_frame = 10000
# image_folder_name = '0520_2036_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2036_black_four.mov'
# start_frame = 10000
# end_frame = 20000
# image_folder_name = '0520_2036_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2008_black_one.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0520_2008_black_one_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 1
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2040_black_one.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0520_2040_black_one_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 1
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2042_black_one.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0520_2042_black_one_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 1
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2057_black_one.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0520_2057_black_one_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 1
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0520/Salk_colored_three.mp4'
# start_frame = 0
# end_frame = 300
# image_folder_name = '0520_Salk_colored_three_mp4_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 3
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2109_black_four_mixed_gender.mov'
# start_frame = 0
# end_frame = 10000
# image_folder_name = '0520_2109_black_four_mixed_gender_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 3
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2111_black_four_mixed_gender.mov'
# start_frame = 0
# end_frame = 10000
# image_folder_name = '0520_2111_black_four_mixed_gender_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 3
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0520/2109_black_four_mixed_gender.mov'
# start_frame = 10000
# end_frame = 20000
# image_folder_name = '0520_2109_black_four_mixed_gender_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 3
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0521/1924_black_four_male.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_1924_black_four_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0521/1953_black_four_male.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_1953_black_four_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0521/2022_black_four_male.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_2022_black_four_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0521/2022_black_four_male.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_2022_black_four_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0521/2040_black_four_male.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_2040_black_four_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0521/2105_black_one_male.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_2105_black_one_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 1
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0521/2120_black_two_male.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_2120_black_two_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 2
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0521/2144_black_four_mice.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0521_2144_black_four_mice_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0521/2022_black_four_male.mov'
# start_frame = 15857-30*10
# end_frame = 15947+30*10
# image_folder_name = '0521_2022_black_four_male_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0525/1854_black_five.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0525_1854_black_five_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 5
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0525/1909_black_five.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0525_1909_black_five_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 5
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0525/1838_black_five.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0525_1838_black_five_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 5
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0525/1901_black_five.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0525_1901_black_five_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 5
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0525/1915_black_five.mov'
# start_frame = 0
# end_frame = 500
# image_folder_name = '0525_1915_black_five_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 5
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0527/1210_black_four.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0527_1210_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0527/1012_black_four.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0527_1012_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0527/1128_black_four.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0527_1128_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0531/1126_black_four.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0531_1126_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0531/1108_black_four.mov'
# start_frame = 0
# end_frame = 50000
# image_folder_name = '0531_1108_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0531/1108_black_four.mov'
# start_frame = 0
# end_frame = 500
# image_folder_name = '0531_1108_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1954_black_four.mov'
# start_frame = 0
# end_frame = 18000
# image_folder_name = '0601_1954_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1100_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_1100_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1844_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_1844_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0601/2009_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_2009_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1056_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_1056_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1143_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_1143_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1130_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_1130_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1212_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_1212_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1954_black_four.mov'
# start_frame = 0
# end_frame = 18000
# image_folder_name = '0601_1954_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1922_black_four.mov'
# start_frame = 0
# end_frame = 60000
# image_folder_name = '0601_1922_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1954_black_four.mov'
# start_frame = 18000
# end_frame = 36000
# image_folder_name = '0601_1954_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"


# video_full_path = '/disk4/zexin/project/mice/datasets/0601/1954_black_four.mov'
# start_frame = 36000
# end_frame = 60000
# image_folder_name = '0601_1954_black_four_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 4
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"



video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1523_two_black.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0518_1523_two_black_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1533_two_black.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0518_1533_two_black_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0518/Black Two/1353_two_black.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0518_1353_two_black_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0603/1022_black_two.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0603_1022_black_two_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0603/0825_black_two.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0603_0825_black_two_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0603/0808_black_two.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0603_0808_black_two_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0603/1001_black_two.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0603_1001_black_two_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0603/1115_black_two.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0603_1115_black_two_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

video_full_path = '/disk4/zexin/project/mice/datasets/0603/1038_black_two.mov'
start_frame = 0
end_frame = 60000
image_folder_name = '0603_1038_black_two_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"


video_full_path = '/disk4/zexin/project/mice/datasets/0603/1258_black_two.mov'
start_frame = 0
end_frame = 80000
image_folder_name = '0603_1258_black_two_mov_%d_%d'%(start_frame,end_frame)
max_pid_id_setting = 2
match = 0
weights = '0 6 0 0 0 0 '
video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0603/1312_black_two.mov'
# start_frame = 0
# end_frame = 80000
# image_folder_name = '0603_1312_black_two_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 2
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0603/1323_black_two.mov'
# start_frame = 0
# end_frame = 80000
# image_folder_name = '0603_1323_black_two_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 2
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"

# video_full_path = '/disk4/zexin/project/mice/datasets/0603/1959_black_two.mov'
# start_frame = 0
# end_frame = 80000
# image_folder_name = '0603_1959_black_two_mov_%d_%d'%(start_frame,end_frame)
# max_pid_id_setting = 2
# match = 0
# weights = '0 6 0 0 0 0 '
# video_image_save_path = '/disk4/zexin/project/mice/datasets/imageFromVideo/%s/'%(image_folder_name)
# save_image_format = image_folder_name+"_%d.png"



#########################################################################################################
###                                  code path setting                                                ###
#########################################################################################################
alphaPose_root = '/home/zexin/project/mice/AlphaPose/'
darknet_root= alphaPose_root+'/train_yolo/darknet/'
sppe_root = alphaPose_root+'/train_sppe/'


#########################################################################################################
###                                 data related setting                                              ###
#########################################################################################################

# # new dataset labeled by company dataset 0416/01
# image_root_list=['/home/zexin/datasets/mice/new_labeled_byCompany/01/color/']
# json_file_list = ['/disk4/zexin/datasets/mice/new_labeled_byCompany/01/multi_person_01.json']
# exp_name = 'labeled_byCompany_01_split90_ori'
# num_mouse = 4
# num_pose = 4
# train_val_split = 0.90


# # new dataset labeled by company dataset 0416/02
# image_root_list=['/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/']
# json_file_list = ['/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json']
# exp_name = 'labeled_byCompany_02_split90_ori'
# num_mouse = 4
# num_pose = 4
# train_val_split = 0.90

# # new dataset labeled by company dataset 0416/03
# image_root_list=['/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/color/']
# json_file_list = ['/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/annotation_03.json']
# exp_name = 'labeled_byCompany_03_split90_ori'
# num_mouse = 2
# num_pose = 4
# train_val_split = 0.90

# image_root_list=['/home/zexin/datasets/mice/new_labeled_byCompany/01/color/','/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/']
# json_file_list = ['/disk4/zexin/datasets/mice/new_labeled_byCompany/01/multi_person_01.json','/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json']
# exp_name = 'labeled_byCompany_01_02_split90_ori'
# num_mouse = 4
# num_pose = 4
# train_val_split = 0.90

# image_root_list=[\
#                  '/home/zexin/datasets/mice/new_labeled_byCompany/01/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/color/'
#                 ]
# json_file_list = [\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/01/multi_person_01.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/annotation_03.json'
#                  ]
# num_mouse = [4,4,2]
# exp_name = 'labeled_byCompany_01_02_03_split90_ori'
# num_pose = 4
# train_val_split = 0.90


# image_root_list=[\
#                  '/home/zexin/datasets/mice/new_labeled_byCompany/01/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/color/', \
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/'
#                 ]
# json_file_list = [\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/01/multi_person_01.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/annotation_03.json', \
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json'
#                  ]
# num_mouse = [4,4,2,4]
# exp_name = 'labeled_byCompany_01to04_split90_ori'
# num_pose = 4
# train_val_split = 0.90



# image_root_list=[\
#                  '/home/zexin/datasets/mice/new_labeled_byCompany/01/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/color/', \
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/', \
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05/'
#                 ]
# json_file_list = [\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/01/multi_person_01.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/annotation_03.json', \
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/multi_person_05.json'
#                  ]
# num_mouse = [4,4,2,4,4]
# exp_name = 'labeled_byCompany_01to05_split90_ori'
# num_pose = 4
# train_val_split = 0.90


# image_root_list=[\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/', \
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05/'
#                 ]
# json_file_list = [\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/multi_person_05.json'
#                  ]
# num_mouse = [4,4,4]
# exp_name = 'labeled_byCompany_020405_split90_ori'
# num_pose = 4
# train_val_split = 0.90


# image_root_list=[\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/', \
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05/',\
#                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/select_track_frame_merged/'
#                 ]
# json_file_list = [\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/multi_person_05.json',\
#                   '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/multi_person_06.json'
#                  ]
# num_mouse = [4,4,4,4]
# exp_name = 'labeled_byCompany_02040506_split90_ori'
# num_pose = 4
# train_val_split = 0.90


image_root_list=[\
                 '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/color/',\
                 '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04/', \
                 '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05/',\
                 '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/select_track_frame_merged/', \
                 '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/select_track_frame_merge/'
                ]
json_file_list = [\
                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/02/annotation_02.json',\
                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/multi_person_04.json',\
                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/multi_person_05.json',\
                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/06/multi_person_06.json', \
                  '/disk4/zexin/datasets/mice/new_labeled_byCompany/07/multi_person_07.json'
                 ]
num_mouse = [4,4,4,4,5]
exp_name = 'labeled_byCompany_0204050607_split90_ori'
num_pose = 4
train_val_split = 0.90


#########################################################################################################
###                                    automatic setting                                              ###
#########################################################################################################
## general data setting
ln_image_dir = alphaPose_root + '/data/mice/'+exp_name+'/color_image/'

### sppe data setting
train_h5_file = sppe_root+ '/data/mice/'+exp_name+'/data_newLabeled_01_train.h5'
val_h5_file = sppe_root+ '/data/mice/'+exp_name+'/data_newLabeled_01_val.h5'

### yolo data setting
image_suffix = 'png'

color_img_prefix = 'data/mice/'+exp_name+'/color/'
file_list_root = 'data/mice/'+exp_name+'/'

# newdir=darknet_root + '/'+ color_img_prefix
yolo_image_annot_root =darknet_root + '/'+ color_img_prefix
train_list_file = darknet_root+'/' + file_list_root + '/' + 'train.txt'
val_list_file = darknet_root+'/' + file_list_root + '/' + 'valid.txt'

valid_image_root = darknet_root+ '/data/mice/'+exp_name+'/valid_image/'

## evalation setting
gt_json_file_train = alphaPose_root + '/data/mice/'+exp_name+'_gt_forEval_train.json'
gt_json_file_valid = alphaPose_root + '/data/mice/'+exp_name+'_gt_forEval_valid.json'

if not os.path.exists(sppe_root+ '/data/mice/'+exp_name):
    os.mkdir(sppe_root+ '/data/mice/'+exp_name)
if not os.path.exists(darknet_root+ '/data/mice/'+exp_name):
    os.mkdir(darknet_root+ '/data/mice/'+exp_name)
if not os.path.exists(darknet_root+ '/data/mice/'+exp_name+'/color/'):
    os.mkdir(darknet_root+ '/data/mice/'+exp_name+'/color/')
if not os.path.exists(valid_image_root):
    os.mkdir(valid_image_root)
if not os.path.exists(alphaPose_root + '/data/mice/'+exp_name):
    os.mkdir(alphaPose_root + '/data/mice/'+exp_name)
if not os.path.exists(ln_image_dir):
    os.mkdir(ln_image_dir)






#########################################################################################################
###                                              running                                              ###
#########################################################################################################

# ## video_image_save_path = alphaPose_root+'/examples/'+exp_name+'_oriFrameFromVideo'
# if not os.path.exists(video_image_save_path):
#     os.mkdir(video_image_save_path)
# else:
#     os.system('rm {}/*'.format(video_image_save_path))
    
# cap = cv2.VideoCapture(video_full_path)
# if cap.isOpened():
#     success = True
# else:
#     success = False
#     print(" read failed!make sure that the video format is supported by cv2.VideoCapture")

# # while(success):
# for frame_index in tqdm(range(end_frame)):
#     success, frame = cap.read()
#     if not success:
#         print('read frame failed!')
#         break
#     if frame_index < start_frame:
#         continue
#     cv2.imwrite(video_image_save_path +  save_image_format % frame_index, frame)
    
# cap.release()


import os 
import subprocess
from subprocess import Popen

print('getting demo image:')
os.system('cd {}'.format(alphaPose_root))
# subprocess.run('conda activate poseflow')
# subprocess.run("bash -c 'conda activate poseflow'", shell=True)
os.system('CUDA_VISIBLE_DEVICES=\'4\' python3 demo.py \\\n \
--indir {} \\\n \
--outdir examples/{}/  \\\n \
--yolo_model_path {}/backup/{}/yolov3-mice.backup \\\n \
--yolo_model_cfg /disk4/zexin/project/mice/AlphaPose/train_yolo/darknet//cfg/yolov3-mice.cfg \\\n \
--pose_model_path {}exp/coco/{}/model_30.pkl \\\n \
--use_boxGT 0'.format(video_image_save_path,exp_name+'_'+image_folder_name+'_fromVideo',darknet_root,exp_name,sppe_root,exp_name))


print('')
if max_pid_id_setting==1:
    print('there is only one mouse, no need to do the tracking')
else:
    print('tracking pose:')
    # print('use http://localhost:21600/notebooks/project/mice/PoseFlow/tracker-general.ipynb')
    # print('self.imgdir = \'{}\' \n \
    # self.in_json = \'{}/examples/{}/alphapose-results.json\' \n \
    # self.out_json = \'{}/examples/{}/alphapose-results-forvis-tracked.json\' \n \
    # self.visdir = \'{}examples/{}/pose_track_vis/\'  \n \
    # self.image_format = \'{}\'  \n \
    # '.format(video_image_save_path,alphaPose_root,exp_name+'_'+image_folder_name+'_fromVideo',alphaPose_root,exp_name+'_'+image_folder_name+'_fromVideo',alphaPose_root,exp_name+'_'+image_folder_name+'_fromVideo',save_image_format))

    # print('or run the following cmd:')
    # print('cd {}/PoseFlow'.format(alphaPose_root))
    os.system('python ./PoseFlow/tracker-general-fixNum-newSelect-noOrb.py \\\n \
        --imgdir {} \\\n \
        --in_json {}/examples/{}/alphapose-results.json \\\n \
        --out_json {}/examples/{}/alphapose-results-forvis-tracked.json \\\n \
        --visdir {}/examples/{}/pose_track_vis/ \\\n \
        --image_format {} \
        --max_pid_id_setting {} --match {}  --weights {} \\\n \
        --out_video_path {}/examples/{}/{}_{}_{}_{}.mp4  \
        '.format(video_image_save_path,\
        	alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',\
        	alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',\
        	alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',\
        	'%s.png',\
        	max_pid_id_setting, match, weights, \
        	alphaPose_root, exp_name+'_'+image_folder_name+'_fromVideo',exp_name+'_'+image_folder_name+'_fromVideo',max_pid_id_setting, match, weights.replace(' ', '')))
    #     '.format(video_image_save_path,exp_name+'_'+image_folder_name+'_fromVideo',exp_name+'_'+image_folder_name+'_fromVideo',exp_name+'_'+image_folder_name+'_fromVideo',save_image_format))


# os.system('rm -r {}'.format(video_image_save_path))

