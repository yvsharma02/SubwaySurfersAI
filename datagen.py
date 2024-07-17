from recorder import ScreenRecorder
from input_manager import InputManager
from common import Action
import datetime
import keyboard

recorder = ScreenRecorder('Android Emulator - Pixel_4a_API_33:5554', xoffset=0, yoffset=0, height_extension=0, width_extension=0)
input_manager = InputManager('emulator-5554')

def update(count):
#   recorder.save('data\\' + str(c) + '.png')
    give_action()
    
def give_action():
    if keyboard.is_pressed('up'):
        input_manager.give_input(Action.SWIPE_UP)
    elif keyboard.is_pressed('down'):
        input_manager.give_input(Action.SWIPE_DOWN)
    elif keyboard.is_pressed('left'):
        input_manager.give_input(Action.SWIPE_LEFT)
    elif keyboard.is_pressed('right'):
        input_manager.give_input(Action.SWIPE_RIGHT)





c = 0
last_sec_c = 0
last_sec_time = datetime.datetime.now()
while True:
    update(c)
    c += 1
    delay = (datetime.datetime.now() - last_sec_time).seconds
    if (delay >= 1):
        last_sec_time = datetime.datetime.now()
        print("FPS: " + str((c - last_sec_c) / delay))
        last_c = c

    if keyboard.is_pressed('q'):
        break
    

recorder.flush()