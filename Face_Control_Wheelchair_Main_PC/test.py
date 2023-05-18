from ultralytics import YOLO
import camera as cam
import cv2

model = YOLO('C:/Users/user/Desktop/Face_Control_Wheelchair/Traffic_Light/230515_1640_best.pt')

cam = cam.libcamera()
ch0, ch1 = cam.initial_setting(capnum=2)

while True:
    _, frame0, _, frame1 = cam.camera_read(ch0, ch1)
    results = model(frame0)
    result = results[0].plot()

    cv2.imshow('results', result)
    cv2.waitKey(1)