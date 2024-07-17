from recorder import ScreenRecorder
import time
import datetime
recorder = ScreenRecorder('Android Emulator - Pixel_3a_API_30:5554', xoffset=25, yoffset=25, height_extension=150, width_extension=100)

c = 0
last_c = 0
last_time = datetime.datetime.now()
while True:
    recorder.save('test\\' + str(c) + '.png')
    c += 1
    delay = (datetime.datetime.now() - last_time).seconds
    if (delay >= 1):
        last_time = datetime.datetime.now()
        print("FPS: " + str((c - last_c) / delay))
        last_c = c

recorder.flush()