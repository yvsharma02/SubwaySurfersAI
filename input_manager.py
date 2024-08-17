from common import Action
from recorder import ScreenRecorder
import os
import datetime
import random
import config
import numpy as np
from PIL import Image
from subprocess import Popen, PIPE
from sys import stdout

class InputManager:
    input_cmd_map = {
        Action.SWIPE_UP: 'adb shell input touchscreen swipe 500 500 500 200 100',
        Action.SWIPE_DOWN: 'adb shell input touchscreen swipe 500 200 500 500 100',
        Action.SWIPE_LEFT: 'adb shell input touchscreen swipe 500 500 200 500 100',
        Action.SWIPE_RIGHT: 'adb shell input touchscreen swipe 500 500 800 500 100',
        Action.DO_NOTHING: ''
    }

    screen_recorder = None
    shell = None
    out = None

    def __init__(self, screen_recorder):
        self.screen_recorder = screen_recorder
        self.out = open(config.STD_OUT_FOR_SUBPROCESS, "w")
        self.shell = Popen(['cmd.exe'], stdin=PIPE, stdout=self.out, text=True)


    def perform_action(self, action, count = -1, captures_dir = "", record = False):
        saved = False

#        print(config.NOTHING_SKIP_RATE)
        image = None
        if (record):
            image = self.screen_recorder.capture()
            
        cmd = self.input_cmd_map[action]
        
        if (cmd):
#            print("Performing: {}".format(cmd))
            self.shell.stdin.write("{}\n".format(cmd))
            self.shell.stdin.flush()
            print("Performed: {}".format(cmd))

        if (image is not None):

            with open (os.path.join(captures_dir, "commands.txt"), "a") as file:
                file.write(str(count) + "; " + str(int(action)) + "; " + str(datetime.datetime.now()) + ";\n")
            Image.fromarray(image).save(os.path.join(captures_dir, str(count) + ".png"))
            print("Recorded: " + str(int(action)) + " with count: " + str(count))
            saved = True

        if (record and not saved):
            print("Failed to save.")

        return saved

    def flush(self):
        self.shell.kill()
        self.out.close()
#        self.shell.wait()
