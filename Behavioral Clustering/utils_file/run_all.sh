echo 'remenber to change setting in setting.py'
echo 'python 0_video2image.py'
python 0_video2image.py
echo 'python 1_get_video_contour.py'
python 1_get_video_contour.py
# python 1_2_get_video_contour_includeOtherMouse.py
# cd utils_file && python 1_get_video_contour.py 4
echo 'python 2_create_dataframe.py'
python 2_create_dataframe.py
echo 'python 3_pca_onDataframe.py'
python 3_pca_onDataframe.py
# python 2_create_dataframe.py && python 3_pca_onDataframe.py
