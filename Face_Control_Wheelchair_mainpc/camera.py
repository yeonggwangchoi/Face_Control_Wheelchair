import cv2
import numpy as np
import dlib
import time

MID_X = 160

class libcamera(object):
    def __init__(self):
        self.capnum = 0
        self.mid_x = 0
        self.mid_y = 0
        self.x_63 = 0
        self.y_63 = 0
        self.x_67 = 0
        self.y_67 = 0
        self.predictor = dlib.shape_predictor("/home/pi/Face_Control_Wheelchair/Face_Control_Wheelchair_mainpc/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

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
            channel0 = cv2.VideoCapture(cam0port, cv2.CAP_V4L2)
            if channel0.isOpened():
                print("Camera Channel0 is enabled!")
        elif capnum == 2:
            channel0 = cv2.VideoCapture(cam0port, cv2.CAP_V4L2)
            if channel0.isOpened():
                print("Camera Channel0 is enabled!")
            channel1 = cv2.VideoCapture(cam1port, cv2.CAP_V4L2)
            if channel1.isOpened():
                print("Camera Channel0 is enabled!")

        channel0.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        channel0.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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
            print("----------------------------")
            print("얼굴 인식 실패")
            print("----------------------------")
        else:
            print("----------------------------")
            print("얼굴 인식 성공")
            print("----------------------------")
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
                    
                    if i == 63:
                        self.x_63, self.y_63 = shape_point.x, shape_point.y
                    
                    if i == 67:
                        self.x_67, self.y_67 = shape_point.x, shape_point.y
        cv2.line(replica, (160, 0), (160, 240),(0, 0, 255), 3)
        cv2.imshow("Face_Detect", replica)
    
    def face_direction(self):
        if self.y_67 - self.y_63 >= 15:
            print("----------------------------")
            print("비상 정지")
            print("----------------------------")
            return 'S'
        else:
            if abs(self.mid_x - MID_X) <= 20:
                print("----------------------------")
                print("예상 얼굴 방향 : G")
                print("----------------------------")
                return 'G'
            else:
                if MID_X - self.mid_x > 0:
                    abs(MID_X - self.mid_x)
                    print("----------------------------")
                    print("예상 얼굴 방향 : R")
                    print("----------------------------")
                    return 'R'
                else:
                    abs(MID_X - self.mid_x)
                    print("----------------------------")
                    print("예상 얼굴 방향 : L")
                    print("----------------------------")
                    return 'L'