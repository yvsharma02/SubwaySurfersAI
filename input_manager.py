from common import Action
from recorder import ScreenRecorder
import os
import datetime
import random
import config

class InputManager:
    input_cmd_map = {
        Action.SWIPE_UP: 'adb shell input touchscreen swipe 500 500 500 200 100',
        Action.SWIPE_DOWN: 'adb shell input touchscreen swipe 500 200 500 500 100',
        Action.SWIPE_LEFT: 'adb shell input touchscreen swipe 500 500 200 500 100',
        Action.SWIPE_RIGHT: 'adb shell input touchscreen swipe 500 500 800 500 100',
        Action.DO_NOTHING: ''
    }

    screen_recorder = None

    def __init__(self, screen_recorder):
        self.screen_recorder = screen_recorder

    def perform_action(self, action, count = -1, captures_dir = "", record = False):
        saved = False

#        print(config.NOTHING_SKIP_RATE)
        image = None
        if (record):
            if(action != Action.DO_NOTHING or random.random() >= config.NOTHING_SKIP_RATE):
                image = self.screen_recorder.capture()
                # if(self.screen_recorder.save(os.path.join(captures_dir, str(count) + ".png"))):
                #     with open (os.path.join(captures_dir, "commands.txt"), "a") as file:
                #         file.write(str(count) + "; " + str(action) + "; " + str(datetime.datetime.now()) + ";\n")
                    
                #     print("Recorded: " + str(action))
                #     saved = True
        cmd = self.input_cmd_map[action]
        
        if (cmd):
            os.system(cmd)

        if (image != None):

            with open (os.path.join(captures_dir, "commands.txt"), "a") as file:
                file.write(str(count) + "; " + str(int(action)) + "; " + str(datetime.datetime.now()) + ";\n")
            image.save(os.path.join(captures_dir, str(count) + ".png"))
            print("Recorded: " + str(int(action)) + " with count: " + str(count))
            saved = True

        return saved
