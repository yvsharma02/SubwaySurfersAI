from enum import IntEnum

import tensorflow as tf
import numpy as np

import os
import datetime
import cv2
import random

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

    data : list[tuple] = None

    def mapper(data, label):
        def process_path(path):
            raw = tf.io.read_file(path)
            tensor = tf.io.decode_image(raw)
            tensor = tf.cast(tensor, tf.float32) / 255.0
            tensor = tf.reshape(tensor, shape=config.TRAINING_IMAGE_DIMENSIONS)
            return tensor
        
        images = tf.map_fn(process_path, data, dtype=tf.float32)
        
        return images, label


    # def mapper(data, label):

    #     images = []
    #     for path in data:
    #         raw = tf.io.read_file(path)
    #         tensor = tf.io.decode_image(raw)
    #         tensor = tf.cast(tensor, tf.float32) / 255.0
    #         tensor = tf.reshape(tensor=tensor, shape=config.TRAINING_IMAGE_DIMENSIONS)
    #         images.append(tensor)
        
    #     return tf.stack(images, axis=0), label

    def __init__(self, dir) -> None:

        self.data = []

        if (not dir):
            return

#        print("Reading Dataset: " + dir)
        with open(os.path.join(dir, "commands.txt")) as commands:
            lines = commands.readlines()
#            print(len(lines))
            for line in lines:
                if (len(line.split(';')) == 4):
                    index, action, time, nl = line.split(';')
                    action = action.lstrip().rstrip()
                    time = time.lstrip().rstrip()
                    self.data.append((os.path.join(dir, str(index) + ".png"), int(action), datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")))
                else:
#                    print("Invalid Dataset: " + line)
                    self.data = []

    def count(self):
        return len(self.data)

    # Maybe any built in library has a better way to do this?
    def remove_samples(self, action, keep_percent):
        self.data = [ds for ds in self.data if ds[1] != action or random.random() < keep_percent]

    def multiply_samples(self, action, multipler):
        all = [ds for ds in self.data if ds[1] == action]
        count = int(len(all) * (multipler - 1.0))
        self.data.extend([all[int(random.random() * (len(all) - 1))] for i in range(0, count)])

    def show_img(self, ind, waittime):
        cv2.imshow("image", self.data[ind])
        cv2.waitKey(waittime)

    def summary(self):
        c = 0
        example_labels = {}
        for data_point in self.data:
            label = Action(data_point[1])
            if (label in example_labels):
                example_labels[label] += 1
            else:
                example_labels[label] = 1
#        print(example_labels)

    def get_dataset(self, only_nothing_skip_rate) -> tf.data.Dataset:

        training_data = []
        labels = []


        for i in range(0, len(self.data) - (config.SEQUENCE_LEN - 1)):
            img_list = [self.data[x][0] for x in range(i, i + config.SEQUENCE_LEN)]
            # all_nothing = True
            # for x in range(i, i + config.SEQUENCE_LEN):
            #     if (self.data[x][1] != int(Action.DO_NOTHING)):
            #         all_nothing = False
            #         break

            all_nothing = self.data[i + config.SEQUENCE_LEN - 1][1] == int(Action.DO_NOTHING)

            if (not all_nothing or random.random() > only_nothing_skip_rate):
                training_data.append(img_list)
                labels.append(self.data[i + config.SEQUENCE_LEN - 1][1])

#        print(training_data)
        dataset = tf.data.Dataset.from_tensor_slices((training_data, labels))
        dataset = dataset.map(CustomDataSet.mapper, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

def combine_custom_datasets(datasets : list[CustomDataSet]):
    ds = CustomDataSet(None)
    ds.data = [data_point for cds in datasets for data_point in cds.data]
    return ds