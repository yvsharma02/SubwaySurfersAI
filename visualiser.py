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

import cv2
import tensorflow as tf
#from keras import models, layers, losses


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

visualise(3, 'generated/visualiser/loose-3C95-layer-1', vis_CNN3, 'generated/output/3C95/player/loose-3C95/2024-08-22-20-22-22')
#vis_CNN3(3, 'test', CNN3)

# conv_layers = [i for i in range(0, len(model.layers)) if model.layers[i].name.startswith("conv2d")]
# outputs = [model.layers[i].output for i in conv_layers]
# model_vis = keras.Model(inputs=model.inputs, outputs=outputs)

# def im_loader(path):
#     raw = tf.io.read_file(path)
#     tensor = tf.io.decode_image(raw)
#     tensor = tf.cast(tensor, tf.float32) / 255.0
#     tensor = tf.reshape(tensor=tensor, shape=(1, 172, 80, 3))
#     return tensor

# im = im_loader("generated/data/downscaled/2024-07-26-23-4-4/45.png")
# print(im.shape)

# out = model_vis.predict(im)

# layer_count = 0
# for layer in out:
#     layer_out = tf.squeeze(layer, axis = 0)
#     images = tf.unstack(layer_out, axis = -1)

#     im_count = 0
#     for im in images:
#         path = "generated/visualiser/{layerno}/Filter-{filterno}".format(layerno=layer_count, filterno=im_count) + ".png"
#         cv2.imwrite(path, 255*im.numpy())
#         im_count += 1
#     layer_count += 1