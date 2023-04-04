import cv2
import numpy as np
import dlib
import torch
import time
from ultralytics import YOLO
MID_X = 320

class libcamera(object):
    def __init__(self):
        self.capnum = 0
        self.mid_x = 0
        self.mid_y = 0
        self.predictor = dlib.shape_predictor("Face_Control_Wheelchair/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.traffic_light_model = YOLO('yolov8n.pt')
        self.obstacle_model = YOLO('yolov8n-seg.pt')
    def loop_break(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Camera Readding is ended.")
            return True
        else:
            return False
    
    def initial_setting(self, cam0port=0, cam1port=1, capnum=1):
        print("OpenCV Version : ", cv2.__version__)
        channel0 = None
        channel1 = None
        self.capnum = capnum

        if capnum == 1:
            channel0 = cv2.VideoCapture(cam0port, cv2.CAP_DSHOW)
            if channel0.isOpened():
                print("Camera Channel0 is enabled!")
        elif capnum == 2:
            channel0 = cv2.VideoCapture(cam0port, cv2.CAP_DSHOW)
            if channel0.isOpened():
                print("Camera Channel0 is enabled!")
            channel1 = cv2.VideoCapture(cam1port, cv2.CAP_DSHOW)
            if channel1.isOpened():
                print("Camera Channel0 is enabled!")

        channel0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        channel0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        channel0.set(cv2.CAP_PROP_FPS, 60)
        
        channel1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        channel1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        channel1.set(cv2.CAP_PROP_FPS, 60)

        return channel0, channel1
    
    def camera_read(self, cap1, cap2 = None):
        result, capset = [], [cap1, cap2]

        for idx in range(0, self.capnum):
            ret, frame = capset[idx].read()
            result.extend([ret, frame])

        return result
    
    def image_show(self, frame0, frame1=None):
        if frame1 is None:
            cv2.imshow('frame0', frame0)
        else:
            cv2.imshow('frame0', frame0)
            cv2.imshow('frame1', frame1)
        
    def face_detect(self, frame0):
        detector = self.detector
        predictor = self.predictor

        replica = frame0.copy()
        rows, cols = replica.shape[:2]
        dets = detector(replica, 1)

        if len(dets) == 0:
            print(False)
        else:
            print(True)
            for k, d in enumerate(dets):
                shape = predictor(replica, d)
                cv2.rectangle(replica, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 3)
                
                for i in range(shape.num_parts):
                    shape_point = shape.part(i)
                    
                    if i < 17:
                        cv2.circle(replica, (shape_point.x, shape_point.y), 1, (255,0,0), 1)
                    else:
                        cv2.circle(replica, (shape_point.x, shape_point.y), 1, (0,255,0), 1)

                    if i == 34:
                        self.mid_x, self.mid_y = shape_point.x, shape_point.y
        
        self.image_show(replica)
    
    def face_direction(self):

        if abs(self.mid_x - MID_X) < 20:
            return 'G'
        else:
            if MID_X - self.mid_x > 0:
                abs(MID_X - self.mid_x)
                return 'R'
            else:
                abs(MID_X - self.mid_x)
                return 'L'
            
    def traffic_light(self, frame):
        results = self.traffic_light_model(frame)
        result = results[0].plot()
        cv2.imshow("YOLOv8_traffic_light", result)
            
    def obstacle(self, frame):
        results = self.obstacle_model(frame)
        result = results[0].plot()
        cv2.imshow("YOLOv8_obstacle", result)