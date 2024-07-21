from enum import IntEnum

import tensorflow as tf
import numpy as np

import os
import datetime
import cv2

import config

def date_to_dirname(run_start_time):
    return str(run_start_time.date()) + "-" + str(run_start_time.hour) + "-" + str(run_start_time.minute) + "-" + str(run_start_time.minute)

class Action(IntEnum):
    SWIPE_UP = 0,
    SWIPE_DOWN = 1,
    SWIPE_LEFT = 2,
    SWIPE_RIGHT = 3
    DO_NOTHING = 4

class CustomDataSet:

    data : list[tuple] = []

    def im_loader(path, label):
        raw = tf.io.read_file(path)
        tensor = tf.io.decode_image(raw)
        tensor = tf.cast(tensor, tf.float32) / 255.0
        tensor = tf.reshape(tensor=tensor, shape=config.FINAL_IMAGE_SHAPE)
        return tensor, label

    def __init__(self, dir) -> None:

        if (not dir):
            return

        print("Reading Dataset: " + dir)
        with open(os.path.join(dir, "commands.txt")) as commands:
            lines = commands.readlines()
            for line in lines:
                if (len(line.split(';')) == 4):
                    index, action, time, nl = line.split(';')
                    action = action.lstrip().rstrip()
                    time = time.lstrip().rstrip()
#                    im = self.load_image(os.path.join(self.dir, str(index) + ".png"))
                    self.data.append((os.path.join(dir, str(index) + ".png"), int(action), datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")))
                elif (len(line.split(';')) == 3 or len(line.split(';')) == 2):
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
        cv2.imshow("image", self.data[ind])
        cv2.waitKey(waittime)

    def get_dataset(self) -> tf.data.Dataset:
        labels = [datapoint[1] for datapoint in self.data]
        paths = [datapoint[0] for datapoint in self.data]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(CustomDataSet.im_loader)
        # for x in dataset.take(1):
        #    print(x)
        return dataset

def combine_custom_datasets(datasets : list[CustomDataSet]):
    ds = CustomDataSet(None)
    ds.data = [data_point for cds in datasets for data_point in cds.data]
    return ds