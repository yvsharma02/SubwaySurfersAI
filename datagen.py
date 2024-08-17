import datetime
import config
import keyboard
import common
import os

from input_manager import InputManager
from common import Action
from recorder import ScreenRecorder

recorder = ScreenRecorder(config.SCREEN_NAME)
input_manager = InputManager(recorder)

frame_duration = 1.0 / config.TARGET_FPS
record_count = 0

def fixed_rate_update(count, keypress_cooldown_time_remaining):
    return check_for_actions(count, keypress_cooldown_time_remaining, True)

def immediate_update(count, keypress_cooldown_time_remaining):
    return check_for_actions(count, keypress_cooldown_time_remaining, False)

def keypress_to_action():
    if keyboard.is_pressed('up'):
        return Action.SWIPE_UP
    if keyboard.is_pressed('down'):
        return Action.SWIPE_DOWN
    if keyboard.is_pressed('left'):
        return Action.SWIPE_LEFT
    if keyboard.is_pressed('right'):
        return Action.SWIPE_RIGHT
    else:
        return Action.DO_NOTHING

def check_for_actions(count, keypress_cooldown_time_remaining, record_nothing = False):

    action = keypress_to_action()

    after_keypress_cooldown = keypress_cooldown_time_remaining > config.DATA_GEN_ACTION_PERFORM_COOLDOWN

    if (action == Action.DO_NOTHING):
        if (record_nothing):
            input_manager.perform_action(Action.DO_NOTHING, count, out_dir, True)
            return count + 1, False
    else:
        if (after_keypress_cooldown):
            input_manager.perform_action(action, count, out_dir, True)
            return count + 1, True

    return count, False


print("Press f to start, q to quit, t to stop")

last_frame_time = datetime.datetime.now()
keypress_cooldown_start_time = datetime.datetime.now()

while True:
    
    if (not keyboard.is_pressed('f') and not keyboard.is_pressed('q')):
        continue
    if (keyboard.is_pressed('q')):
        break

    record_count = 0
    
    out_dir = os.path.join(config.ORIGINAL_DATA_DIR, common.date_to_dirname(datetime.datetime.now()))
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    while True:

        current_time = datetime.datetime.now()
        elapsed_time = (current_time - last_frame_time).total_seconds()
        cooldown_time_remaining = (current_time - keypress_cooldown_start_time).total_seconds()

        performed = False

        if cooldown_time_remaining >= config.DATA_GEN_ACTION_PERFORM_COOLDOWN:
            if elapsed_time >= frame_duration:
                record_count, performed = fixed_rate_update(record_count, cooldown_time_remaining)
                last_frame_time = current_time
            else:
                record_count, performed = immediate_update(record_count, cooldown_time_remaining)

        if (performed):
            keypress_cooldown_start_time = datetime.datetime.now()

        if (keyboard.is_pressed('t')):
            break
    
    print("Press f to start, q to quit, t to stop")

input_manager.flush()