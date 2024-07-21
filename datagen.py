from recorder import ScreenRecorder
from input_manager import InputManager
from common import Action
import common
import datetime
import keyboard
import os
import config

#IDLE_ACTION_TIME_MS = 1000

run_start_time = datetime.datetime.now()
recorder = ScreenRecorder(config.SCREEN_NAME)
input_manager = InputManager(recorder)

#def get_time_path(run_start_time):
#    return str(run_start_time.date()) + "-" + str(run_start_time.hour) + "-" + str(run_start_time.minute) + "-" + str(run_start_time.minute)

def get_data_dir(run_start_time):
    res = "data/" + common.date_to_dirname(run_start_time) + "/"
    if (not os.path.exists(res)):
        os.makedirs(res)

    return res

def update(count, last_record_time : datetime.datetime):

    if (last_record_time != None and (datetime.datetime.now() - last_record_time).microseconds / 1000 < config.MIN_DELAY_BETWEEN_ACTIONS_MS):
        return last_record_time

    if (last_record_time == None):
        last_record_time = datetime.datetime.now()
    if keyboard.is_pressed('up'):
        input_manager.perform_action(Action.SWIPE_UP, count, get_data_dir(run_start_time), True)
        last_record_time = datetime.datetime.now()
    elif keyboard.is_pressed('down'):
        input_manager.perform_action(Action.SWIPE_DOWN, count, get_data_dir(run_start_time), True)
        last_record_time = datetime.datetime.now()
    elif keyboard.is_pressed('left'):
        input_manager.perform_action(Action.SWIPE_LEFT, count, get_data_dir(run_start_time), True)
        last_record_time = datetime.datetime.now()
    elif keyboard.is_pressed('right'):
        input_manager.perform_action(Action.SWIPE_RIGHT, count, get_data_dir(run_start_time), True)
        last_record_time = datetime.datetime.now()
    else:
#        diff = (datetime.datetime.now() - last_action_time)
#        time_since_last_action = diff.seconds * 1000 + diff.microseconds / 1000
#        if (time_since_last_action >= IDLE_ACTION_TIME_MS):
        if(input_manager.perform_action(Action.DO_NOTHING, count, get_data_dir(run_start_time), True)):
            last_record_time = datetime.datetime.now()
    
    return last_record_time

frame_counter = 0
last_fps_marker_time = datetime.datetime.now()
last_fps_marker_frame = 0

print("Press any key to start")
keyboard.read_key()

c = 0
last_action_time = None
while True:
    new_action_time = update(c, last_action_time)
    if (new_action_time != last_action_time):
        c += 1
    last_action_time = new_action_time
    frame_counter += 1

    if ((datetime.datetime.now() - last_fps_marker_time).seconds >= 1):
        print("FPS: " + str(frame_counter - last_fps_marker_frame))
        last_fps_marker_time = datetime.datetime.now()
        last_fps_marker_frame = frame_counter

    if keyboard.is_pressed('q'):
        break

recorder.flush()