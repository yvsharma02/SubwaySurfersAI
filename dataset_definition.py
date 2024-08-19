from common import Action

import config
import os
import tensorflow as tf
import random
import global_config
import downscaler

class DatasetDefinition:

    dataset_name : str = None
    nothing_skip_rate = None
    # List of tuple (path_to_dataset, (start_trim_count, end_trim_count), list of specific indexes to ignore, include_mirrored).
    datasets : list[tuple[str, tuple[int, int], list[int], bool]] = None

    loaded_dimensions = None
    # Complete image path, associated action (label)
    data : list[tuple[str, Action]] = None

    # def __init__(self, datasets : list[str, tuple[int, int]] = None):
    #     self.datasets = datasets

    def load(self, height : int, width : int):

        if (self.loaded_dimensions != (height, width)):
            self.reset()

        self.loaded_dimensions = (height, width)

        for i in range(0, len(self.datasets)):
            downscaler.downscale(self.datasets[i][0], height, width, self.datasets[i][3], False)

        complete_dirs = [(os.path.join(global_config.DOWNSCALED_DIR, ds[0]), ds[1]) for ds in self.datasets]
        complete_dirs.append([(os.path.join(global_config.DOWNSCALED_DIR, ds[0] + " - reversed"), ds[1]) for ds in self.datasets if ds[3] == True])
        self.data = []

        for ds in complete_dirs:
            complete_downscaled_dir = os.path.join(ds[0], "commands.txt")
            ignore_indices = ds[2]
            trim_limits = ds[1]
            with open(complete_downscaled_dir) as f:
                lines = f.readlines()[trim_limits[0]:trim_limits[1]]
                self.data.append([
                    (os.path.join(complete_downscaled_dir, "{}.png".format(index)), Action(int(label))) 
                    for line in lines 
                    for index, label, date in line.replace(' ', '').split(';') if index not in ignore_indices])


    def reset(self):
        self.data = []
        self.loaded_dimensions = None

def to_tf_dataset(dataset : DatasetDefinition, sequence_length : int, input_shape : tuple):

    def convert_path_to_image(self, data, label):
        def process_path(path):
            raw = tf.io.read_file(path)
            tensor = tf.io.decode_image(raw)
            tensor = tf.cast(tensor, tf.float32) / 255.0
            tensor = tf.reshape(tensor, shape=input_shape)
            return tensor
        
        images = tf.map_fn(process_path, data, dtype=tf.float32)
        
        return images, label

    data = []
    labels = []

    nothing_indices = [i for i in range(sequence_length, len(data)) if data[i][1] == int(Action.DO_NOTHING)]

    keep_count = int(len(nothing_indices) * (1.0 - dataset.nothing_skip_rate))
    keep_indices = random.sample(nothing_indices, keep_count)

    for i in range(0, len(dataset.data) - (sequence_length - 1)):
        
        last_index = i + sequence_length - 1

        if (dataset.data[last_index][1] == int(Action.DO_NOTHING) and last_index not in keep_indices):
            continue

        img_list = [dataset.data[x][0] for x in range(i, i + sequence_length)]

        data.append(img_list)
        labels.append(dataset.data[last_index][1])

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(convert_path_to_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
