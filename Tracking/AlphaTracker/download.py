from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile



sppe_pretrain_weight = '1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW'
yolo_pretrain_weight = '1g8uJjK7EOlqrUCmjZTtCegwnNsBig6zn'
sppe_trained_weight = '1_BwtYySpX9uWDgdwqw0UEppyMYYv1gkJ'
yolo_trained_weight = '13zXkuZ4dNm3ZOwstr1sSWKOOzJ19XZpN'
demo_data = '1N0JjazqW6JmBheLrn6RoDTSRXSPp1t4K'

gdd.download_file_from_google_drive(file_id=sppe_pretrain_weight,dest_path='./models/sppe/duc_se.pth')
gdd.download_file_from_google_drive(file_id=yolo_pretrain_weight,dest_path='./train_yolo/darknet/darknet53.conv.74')
gdd.download_file_from_google_drive(file_id=sppe_trained_weight,dest_path='./train_sppe/exp/coco/demo/model_10.pkl')
gdd.download_file_from_google_drive(file_id=yolo_trained_weight,dest_path='./train_yolo/darknet/backup/demo/yolov3-mice_final.weights')
gdd.download_file_from_google_drive(file_id=demo_data,dest_path='./data/demo.mp4')


