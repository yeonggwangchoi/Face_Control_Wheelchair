import cv2
import dlib
print(cv2.__version__)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(-1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FPS, 30)

print(cap.get(cv2.CAP_PROP_FPS))
while True:
    retval, frame = cap.read()
    dets = detector(frame, 1)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        print(shape.num_parts)
        print(' ')
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 3)
        
        for i in range(shape.num_parts):
            shape_point = shape.part(i)
            
            if i < 17:
                cv2.circle(frame, (shape_point.x, shape_point.y), 1, (255,0,0), 1)
            else:
                cv2.circle(frame, (shape_point.x, shape_point.y), 1, (0,255,0), 1)
                
    cv2.imshow("img", frame)
    cv2.waitKey(1)
    
    
    
    