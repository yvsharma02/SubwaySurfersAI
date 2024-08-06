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

# def get_data_dir(run_start_time):
#     res = os.path.join(config.ORIGINAL_DATA_DIR, date_to_dirname(run_start_time))
#     if (not os.path.exists(res)):
#         os.makedirs(res)

#     return res

# def get_output_dir(run_start_time):
#     res = os.path.join(config.MODEL_OUTPUT_DIR, date_to_dirname(run_start_time))
#     if (not os.path.exists(res)):
#         os.makedirs(res)

#     return res

class Action(IntEnum):
    SWIPE_UP = 0,
    SWIPE_DOWN = 1,
    SWIPE_LEFT = 2,
    SWIPE_RIGHT = 3
    DO_NOTHING = 4

class CustomDataSet:

    data : list[tuple] = None

    def im_loader(path, label):
        raw = tf.io.read_file(path)
        tensor = tf.io.decode_image(raw)
        tensor = tf.cast(tensor, tf.float32) / 255.0
        tensor = tf.reshape(tensor=tensor, shape=config.TRAINING_IMAGE_DIMENSIONS)
        return tensor, label

    def __init__(self, dir) -> None:

        self.data = []

        if (not dir):
            return

        print("Reading Dataset: " + dir)
        with open(os.path.join(dir, "commands.txt")) as commands:
            lines = commands.readlines()
            print(len(lines))
            for line in lines:
                if (len(line.split(';')) == 4):
                    index, action, time, nl = line.split(';')
                    action = action.lstrip().rstrip()
                    time = time.lstrip().rstrip()
                    self.data.append((os.path.join(dir, str(index) + ".png"), int(action), datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")))
                else:
                    print("Invalid Dataset: " + line)
                    self.data = []

    def count(self):
        return len(self.data)

#Maybe any built in library has a better way to do this?
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
#            print(c)
#            c += 1
#            print(data_point)
#            im = data_point[0]
            label = Action(data_point[1])
            if (label in example_labels):
                example_labels[label] += 1
            else:
                example_labels[label] = 1
        print(example_labels)

    def get_dataset(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        train_indices = [i for i in range(0, len(self.data)) if random.random() >= config.TRAINING_FRACTION]
        test_indices = [i for i in range(0, len(self.data)) if i not in train_indices]
        test_indices = [i for i in test_indices if int(self.data[i][1]) != int(Action.DO_NOTHING)] 
#        train_indices = [i for i in train_indices if int(self.data[i][1]) != int(Action.DO_NOTHING)]        

        # for i in test_indices:
        #     print(Action(self.data[i][1]))

        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]

        labels = [datapoint[1] for datapoint in train_data]
        paths = [datapoint[0] for datapoint in train_data]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(CustomDataSet.im_loader)
        # for x in dataset.take(1):
        #    print(x)

        test_labels = [datapoint[1] for datapoint in test_data]
        test_paths = [datapoint[0] for datapoint in test_data]
        test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
        test_dataset = test_dataset.map(CustomDataSet.im_loader)

        return dataset, test_dataset

def combine_custom_datasets(datasets : list[CustomDataSet]):
    ds = CustomDataSet(None)
    ds.data = [data_point for cds in datasets for data_point in cds.data]
    # print("DSL::")
#    print(len(datasets[0].data))
    return ds