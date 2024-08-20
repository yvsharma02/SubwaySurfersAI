from action import Action

import config
import os
import tensorflow as tf
import random
import global_config
import downscaler

class CustomDataset:

    dataset_name : str = None
    nothing_skip_rate = None
    
    # List of tuple (path_to_dataset, (start_trim_count, end_trim_count), list of specific indexes to ignore, include_mirrored).
    datasets : list[tuple[str, tuple[int, int], list[int], bool]] = None

    currently_loaded_dimensions = None
    # Complete image path, associated action (label)
    path_label_pair : list[tuple[str, Action]] = None

    def load_path_label_pairs(self, height : int, width : int):

        if (self.currently_loaded_dimensions != (height, width)):
            self.reset()

        self.currently_loaded_dimensions = (height, width)
        for i in range(0, len(self.datasets)):
            print("Downscaling: {}/{}".format(i + 1, len(self.datasets)))
            downscaler.downscale(self.datasets[i][0], height, width, self.datasets[i][3], False)

        complete_dirs = [(os.path.join(global_config.DOWNSCALED_DIR, "{}x{}".format(height, width), ds[0]), ds[1], ds[2]) for ds in self.datasets]
        complete_dirs = complete_dirs + [(os.path.join(global_config.DOWNSCALED_DIR, "{}x{}".format(height, width), ds[0] + " - reversed"), ds[1], ds[2]) for ds in self.datasets if ds[3] == True]
        self.path_label_pair = []

        for ds in complete_dirs:
            ignore_indices = ds[2]
            trim_limits = ds[1]
            with open(os.path.join(ds[0], "commands.txt")) as f:
                lines = f.readlines()[trim_limits[0]:trim_limits[1]]

                for line in lines:
                    line = line.replace(' ', '')
                    index, label, date, nl = line.split(';')
                    
                    if index not in ignore_indices:
                        filepath = os.path.join(ds[0], "{}.png".format(index))
                        self.path_label_pair.append([filepath, int(label)])

                # self.data.append([
                #     (os.path.join(complete_downscaled_dir, "{}.png".format(index)), Action(int(label))) 
                #     for line in lines 
                #     for index, label, date, nl in line.replace(' ', '').split(';') if index not in ignore_indices])


    def reset(self):
        self.path_label_pair = []
        self.currently_loaded_dimensions = None

def to_tf_dataset(dataset : CustomDataset, sequence_length : int, img_res : tuple) -> tf.data.Dataset:

    def convert_path_to_image(img_path, label):
        def load_img(path):
            raw = tf.io.read_file(path)
            tensor = tf.io.decode_image(raw)
            tensor = tf.cast(tensor, tf.float32) / 255.0
            tensor = tf.reshape(tensor, shape=img_res)
            return tensor
        
        images = tf.map_fn(load_img, img_path, dtype=tf.float32)
        return images, label

    data = []
    labels = []
    nothing_indices = [i for i in range(sequence_length, len(dataset.path_label_pair)) if dataset.path_label_pair[i][1] == int(Action.DO_NOTHING)]

    keep_count = int(len(nothing_indices) * (1.0 - dataset.nothing_skip_rate))
    keep_indices = random.sample(nothing_indices, keep_count)

    for i in range(0, len(dataset.path_label_pair) - (sequence_length - 1)):
        
        last_index = i + sequence_length - 1

        if (dataset.path_label_pair[last_index][1] == int(Action.DO_NOTHING) and last_index not in keep_indices):
            continue

        img_list = [dataset.path_label_pair[x][0] for x in range(i, i + sequence_length)]

        data.append(img_list)
        labels.append(dataset.path_label_pair[last_index][1])

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(convert_path_to_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
