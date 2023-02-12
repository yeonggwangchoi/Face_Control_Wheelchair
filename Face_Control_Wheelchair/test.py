import cv2
import dlib
print(cv2.__version__)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

img = cv2.imread("img.jpg")
dets = detector(img, 1)
img_c = img.copy()
for k, d in enumerate(dets):
    shape = predictor(img, d)
    print(shape.num_parts)
    print(' ')
    cv2.rectangle(img_c, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255), 3)
    
    for i in range(shape.num_parts):
        shape_point = shape.part(i)
        
        if i < 17:
            cv2.circle(img_c, (shape_point.x, shape_point.y), 1, (255,0,0), 1)
        else:
            cv2.circle(img_c, (shape_point.x, shape_point.y), 1, (0,255,0), 1)
            
cv2.imshow("img", img)
cv2.imshow("img_c", img_c)
cv2.waitKey(0)