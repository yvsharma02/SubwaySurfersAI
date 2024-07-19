from enum import Enum

import os

import datetime
import cv2


class Action(Enum):
    SWIPE_UP = 1,
    SWIPE_DOWN = 2,
    SWIPE_LEFT = 3,
    SWIPE_RIGHT = 4
    DO_NOTHING = 5

class Dataset:

    dir = ""
    data = []

    def __init__(self, dir) -> None:
        self.dir = dir
        with open(os.path.join(dir, "commands.txt")) as commands:
            lines = commands.readlines()
            for line in lines:
                if (len(line.split(';')) == 4):
                    index, action, time, nl = line.split(';')
                    action = action.lstrip().rstrip()[7:]
                    time = time.lstrip().rstrip()
                    self.data.append((Action[action], datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")))
                elif (len(line.split(';')) == 3 or len(line.split(';')) == 3):
                    start, end = line.split(';')
                    start = int(start)
                    end = int(end)
                    self.data = self.data[start:end]
                else:
                    print("Invalid Dataset: " + line)
                    self.data = []    

    def count(self):
        return len(self.data)

    def show_img(self, ind, waittime):
        path = os.path.join(self.dir, str(ind) + ".png")
        print(path)
        cv2.imshow("image", cv2.imread(path, cv2.IMREAD_COLOR))
        cv2.waitKey(waittime)