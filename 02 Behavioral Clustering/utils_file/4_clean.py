import os 

import setting


if __name__ == '__main__':
    arg = setting.args_class()

    for video_path, pose_track_vis_path in zip(arg.videodir,arg.imgdir):
        cmd_line = 'rm -r %s'%(pose_track_vis_path)
        print(cmd_line)
        os.system(cmd_line)
        pass
    



