###################################
# this script will curate a tracked json file 
# please note that the injson should be tracked
###################################
import json
import argparse

from utils_kalman import *


parser = argparse.ArgumentParser(description='FoseFlow Tracker')
parser.add_argument('--in_json', type=str, required=True, help="result json predicted by AlphaPose")
parser.add_argument('--out_json', type=str, required=True, help="result json predicted by AlphaPose")
parser.add_argument('--frame_start', type=int, required=True, help="")
parser.add_argument('--frame_end', type=int, required=True, help="")
parser.add_argument('--num_pose', type=int, required=True, help="")
parser.add_argument('--max_pid_id_setting', type=int, required=True, help="number of mice in the video. -1 for unknown.")
parser.add_argument('--fill_blank_with_predict', type=bool, required=False, default=False, help="")
parser.add_argument('--smooth_pose', type=bool, required=False, default=False, help="")

args = parser.parse_args()
args.kalman = True


with open(args.in_json,'r') as f:
    track_forJson = json.load(f)

post_process_tracking(track_forJson,args)
# display_track_forJson(track_forJson,args)

with open(args.out_json, 'w') as fw:
    json.dump(track_forJson,fw)
















