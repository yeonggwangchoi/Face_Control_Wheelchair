import camera as cam
import atmegaserial as atmega
import raspberryserial as raspberry

import numpy as np
#고정상수&고정변수 초기화
EPOCH = 500000
atmega_port = 'com15'
raspberry_port = 'com18'
red_traffic = False
    
if __name__ == "__main__":
    #각 class 초기화
    ser_at = atmega.ATmega128()
    com_at = ser_at.init(port = atmega_port, baudrate = 9600)
    ser_rasp = raspberry.Raspberry()
    com_desk = ser_rasp.init(port = raspberry_port, baudrate = 9600)
    cam = cam.libcamera()
    ch0, ch1 = cam.initial_setting(capnum=2)

    while True:
        _, frame0, _, frame1 = cam.camera_read(ch0, ch1)

        motor = com_desk.read()
    
        if np.any(cam.object_detect_cls == 9) == True:
           red_traffic = cam.traffic_light_detect(frame0)
        obstacle = cam.object_detect(frame0)

        if (red_traffic == True and motor == b'G') or (obstacle == False and motor == b'G'):
            com_at.write(b'S')
        else:
            com_at.write(motor)
    
        if cam.loop_break():
            break