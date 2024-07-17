from common import Action
import os

class InputManager:
    input_cmd_map = {
        Action.SWIPE_UP: 'adb shell input touchscreen swipe 500 500 500 200 100',
        Action.SWIPE_DOWN: 'adb shell input touchscreen swipe 500 200 500 500 100',
        Action.SWIPE_LEFT: 'adb shell input touchscreen swipe 500 500 200 500 100',
        Action.SWIPE_RIGHT: 'adb shell input touchscreen swipe 500 500 800 500 100',
        Action.DO_NOTHING: ''
    }

    emulator_name = None

    def __init__(self, emulator_name):
        self.emulator_name = emulator_name

    def give_input(self, action):
        cmd = self.input_cmd_map[action]
        if (cmd):
            os.system(cmd)
