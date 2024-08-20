from recorder import ScreenRecorder
from input_manager import InputManager
from action import Action
import action
import datetime
import keyboard
import os
import numpy as np
import keras
import configs
import time
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
#from keras import models, layers, losses

model = tf.keras.models.load_model(os.path.join(configs.MODEL_OUTPUT_DIR, configs.PLAY_MODEL, "model.keras"))

conv_layers = [i for i in range(0, len(model.layers)) if model.layers[i].name.startswith("conv2d")]
outputs = [model.layers[i].output for i in conv_layers]
model_vis = keras.Model(inputs=model.inputs, outputs=outputs)

def im_loader(path):
    raw = tf.io.read_file(path)
    tensor = tf.io.decode_image(raw)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    tensor = tf.reshape(tensor=tensor, shape=(1, 172, 80, 3))
    return tensor

im = im_loader("generated/data/downscaled/2024-07-26-23-4-4/45.png")
print(im.shape)
# arr = im#This is your tensor
# arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()

out = model_vis.predict(im)

layer_count = 0
for layer in out:
    layer_out = tf.squeeze(layer, axis = 0)
    images = tf.unstack(layer_out, axis = -1)

    im_count = 0
    for im in images:
        path = "generated/visualiser/{layerno}/Filter-{filterno}".format(layerno=layer_count, filterno=im_count) + ".png"
        cv2.imwrite(path, 255*im.numpy())
        im_count += 1
    layer_count += 1