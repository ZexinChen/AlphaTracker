import numpy as np
import cv2
import time
import datetime

# cap = cv2.VideoCapture(0)
video_full_path='/Users/app/Desktop/all/lab/project/mice/datasets/data/bleached_one/2041_bleached_mouse_3.mov'
cap = cv2.VideoCapture(video_full_path)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2(history=100)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame1 = np.zeros((640,480))
# out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 5.0, np.shape(frame1))
out = cv2.VideoWriter('BackgroundSubtractor.avi',fourcc, 5.0, np.shape(frame1))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
i = -1

while(1):
    i +=1
    ret, frame = cap.read()

    # if i==0:
    # 	start_frame = frame
    # 	i+=1
    # fgmask = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) - cv2.cvtColor(start_frame,cv2.COLOR_RGB2GRAY)


    fgmask = fgbg.apply(frame)

    # print(np.max(frame))
    fgmask = cv2.erode(fgmask,kernel)
    fgmask = cv2.erode(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)

    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.erode(fgmask,kernel)
    fgmask = cv2.erode(fgmask,kernel)
    fgmask = cv2.erode(fgmask,kernel)
    fgmask[fgmask<250]=0

    (_,cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    for c in cnts:
        Area = cv2.contourArea(c)
        if Area < maxArea :
        #if cv2.contourArea(c) < 500:\
         #if cv2.contourArea(c) < 500:
            (x, y, w, h) = (0,0,0,0)
            continue
        else:
            if Area < 1000:
                (x, y, w, h) = (0,0,0,0)
                continue
            else:
                maxArea = Area
                m=c
                (x, y, w, h) = cv2.boundingRect(m)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(frame)

    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30)&0xff
    if k==27 :
        break

# out.release()
cap.release()
cv2.destoryAllWindows()




