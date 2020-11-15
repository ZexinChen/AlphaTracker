set -x
python 0_video2image.py
python 1_get_video_contour.py
python 2_create_dataframe.py
python 3_pca_onDataframe.py
