from recorder import ScreenRecorder
from input_manager import InputManager
from action import Action
import os
import numpy as np
import keras
import configs
import matplotlib.pyplot as plt
from keras import layers
import settings

from typing import Callable
from PIL import Image
import cv2
import tensorflow as tf

def im_loader(path):
    raw = tf.io.read_file(path)
    tensor = tf.io.decode_image(raw)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    tensor = tf.reshape(tensor=tensor, shape=(170, 82, 3))
    return tensor

def get_layer(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return layer
        if isinstance(layer, layers.TimeDistributed):
            if (layer.layer.name == layername):
                return layer.layer
    return None

def vis_CNN3(output_dir, im_list, im_no):
    model = tf.keras.models.load_model(os.path.join(settings.get_model_train_out_dir("3C95"), "model.keras"))

    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                          outputs=model.layers[0].output)
    print(im_list)
    images = tf.convert_to_tensor([[im_loader(x) for x in im_list]])
    print(images.shape)
    res = intermediate_layer_model.predict(images)
    # Shape is (1, 3, 52, 22, 50) At this point.
    res = tf.squeeze(res, axis = 0)
    # Shape is (3, 52, 22, 50)
    res = tf.unstack(res, axis=-1)
    # Shape is [(3, 52, 22)]
    filter_no = 0
    for filter_out in res:
        images = tf.unstack(filter_out, axis=0)

        dir_path = "{}/{}/".format(output_dir, filter_no)
        if (not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        
        path = "{}/{}.png".format(dir_path, im_no)
        cv2.imwrite(path, 255*images[len(images) - 1].numpy())
#        im_no += 1
        filter_no += 1


def visualise(seq_len, output_dir, func, data_dir,):
    dir_content = ["{}.png".format(i + 1) for i in range(0, len(os.listdir(data_dir)) - 1)]
    for i in range(seq_len - 1, len(dir_content) - seq_len + 1):
        images = [os.path.join(data_dir, dir_content[j]) for j in range(i - seq_len + 1, i + 1)]
        func(output_dir, images, i)

def combine_color_channels(red_folder, green_folder, blue_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    n = len(os.listdir(red_folder))
    
    for i in range(1, n + 1):
        red_image_path = os.path.join(red_folder, f"{i}.png")
        green_image_path = os.path.join(green_folder, f"{i}.png")
        blue_image_path = os.path.join(blue_folder, f"{i}.png")

        if os.path.exists(red_image_path) and os.path.exists(green_image_path) and os.path.exists(blue_image_path):
            red_channel = cv2.imread(red_image_path, cv2.IMREAD_GRAYSCALE)
            green_channel = cv2.imread(green_image_path, cv2.IMREAD_GRAYSCALE)
            blue_channel = cv2.imread(blue_image_path, cv2.IMREAD_GRAYSCALE)

            rgb_image = cv2.merge((blue_channel, green_channel, red_channel))

            output_image_path = os.path.join(output_folder, f"{i}.png")
            cv2.imwrite(output_image_path, rgb_image)
        else:
            continue

def make_movie(in_folder, out_path):
    image_files = sorted([f for f in os.listdir(in_folder) if f.endswith('.png')], 
                         key=lambda x: int(x.split('.')[0]))
    
    if not image_files:
        print(f"No PNG images found in {in_folder}")
        return

    first_image = cv2.imread(os.path.join(in_folder, image_files[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(out_path, fourcc, 8, (width, height), isColor=True)

    for image in image_files:
        img_path = os.path.join(in_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


def find_brightest_channels(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    
    channel_brightness = []

    for subfolder in subfolders:
        image_files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
        if image_files:
            image_path = os.path.join(subfolder, image_files[0])
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            brightness = np.mean(img)
            channel_brightness.append((subfolder, brightness))

    sorted_channels = sorted(channel_brightness, key=lambda x: x[1], reverse=True)

    return [channel[0] for channel in sorted_channels[:3]]

def process_brightest_channels(root_folder, output_folder):
    brightest_channels = find_brightest_channels(root_folder)

    os.makedirs(output_folder, exist_ok=True)

    combo_name = "_".join([os.path.basename(folder) for folder in brightest_channels])
    combo_output_folder = os.path.join(output_folder, combo_name)
    os.makedirs(combo_output_folder, exist_ok=True)

    # Flip the red and green channels
    red_channel, green_channel, blue_channel = brightest_channels
    flipped_channels = [green_channel, red_channel, blue_channel]

    # Use the existing combine_color_channels function with flipped channels
    combine_color_channels(*flipped_channels, combo_output_folder)

    # Generate video name based on folder names
    video_name = f"{combo_name}_flipped_RG.mp4"
    video_path = os.path.join(output_folder, video_name)

    # Make movie from the combined images
    make_movie(combo_output_folder, video_path)

    print(f"Processed brightest channels with flipped red and green, video saved in {output_folder}")

# Example usage
root_folder = 'generated/visualiser/loose-3C95-layer-1/'
output_folder = 'generated/visualiser/loose-3C95-layer-1/output'
process_brightest_channels(root_folder, output_folder)