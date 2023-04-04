import camera as cam
import atmegaserial as atmega

#고정상수&고정변수 초기화
EPOCH = 500000
atmega_port = 'com12'

if __name__ == "__main__":
    #각 class 초기화
    ser = atmega.ATmega128()
    com = ser.init(port = atmega_port, baudrate = 9600)
    cam = cam.libcamera()
    ch0, ch1 = cam.initial_setting(capnum=2)

    for i in range(EPOCH):
        _, frame0, _, frame1 = cam.camera_read(ch0, ch1)
        cam.traffic_light(frame0)
        cam.obstacle(frame0)
        cam.face_detect(frame1)
        sorvo = cam.face_direction()
        print(sorvo)
        com.write(sorvo.encode())
        if cam.loop_break():
            break