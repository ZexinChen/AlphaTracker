set -x
python 0_video2image.py
python 1_2_get_video_contour_includeOtherMouse.py
python 2_create_dataframe.py
python 3_pca_onDataframe.py
