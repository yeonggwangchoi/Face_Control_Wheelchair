import library_copy as lib

EPOCH = 500000

if __name__ == "__main__":
    lib = lib.libcamera()

    ch0, ch1 = lib.initial_setting(capnum=2)

    for i in range(EPOCH):
        _, frame0, _, frame1 = lib.camera_read(ch0, ch1)

        lib.face_detect(frame0)
        
        if lib.loop_break():
            break