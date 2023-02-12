import camera as cam
import raspberryserial as raspberry
import numpy as np

#고정상수&고정변수 초기화
raspberry_port = '/dev/ttyAMA0'
    
if __name__ == "__main__":
    #각 class 초기화
    ser = raspberry.Raspberry()
    com = ser.init(port = raspberry_port, baudrate = 9600)
    cam = cam.libcamera()
    ch0, ch1 = cam.initial_setting(capnum=2)

    while True:
        _, frame0, _, frame1 = cam.camera_read(ch0, ch1)

        cam.face_detect(frame0)
        sorvo = cam.face_direction()
        
        com.write(sorvo.encode())

        if cam.loop_break():
            break