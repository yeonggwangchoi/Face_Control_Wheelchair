import camera as cam
import atmegaserial as atmega
import numpy as np
#고정상수&고정변수 초기화
EPOCH = 500000
atmega_port = 'com15'
red_traffic = False
    
if __name__ == "__main__":
    #각 class 초기화
    ser = atmega.ATmega128()
    com = ser.init(port = atmega_port, baudrate = 9600)
    cam = cam.libcamera()
    ch0, ch1 = cam.initial_setting(capnum=2)

    while True:
        _, frame0, _, frame1 = cam.camera_read(ch0, ch1)

        cam.face_detect(frame0)
        sorvo = cam.face_direction()
        if np.any(cam.object_detect_cls == 9) == True:
           red_traffic = cam.traffic_light_detect(frame1)
        obstacle = cam.object_detect(frame1)

        if (red_traffic == True and sorvo == 'G') or (obstacle == False and sorvo == 'G'):
            com.write(b'S')
        else:
            com.write(sorvo.encode())

        if cam.loop_break():
            break