from recorder import ScreenRecorder
from input_manager import InputManager
from common import Action
import common
import datetime
import keyboard
import os
import numpy as np
import keras
import config
import time
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
#from keras import models, layers, losses

model = tf.keras.models.load_model(os.path.join(config.MODEL_OUTPUT_DIR, config.PLAY_MODEL, "model.keras"))

conv_layers = [i for i in range(0, len(model.layers)) if model.layers[i].name.startswith("conv2d")]
outputs = [model.layers[i].output for i in conv_layers]
model_vis = keras.Model(inputs=model.inputs, outputs=outputs)

print(outputs)

def im_loader(path):
    raw = tf.io.read_file(path)
    tensor = tf.io.decode_image(raw)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    tensor = tf.reshape(tensor=tensor, shape=(1, 172, 80, 3))
    return tensor

im = im_loader("generated/data/downscaled/2024-07-26-23-7-7/1.png")

# arr = im#This is your tensor
# arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
# plt.imshow(arr_)
# plt.show()

out = model_vis.predict(im)
print(out.shape)
out = tf.squeeze(out, axis = 0)
print(out.shape)
out = tf.split(out, num_or_size_splits=32, axis=-1)
#print(out[0])

for i in range(0, 32):
#    cv2.imshow("im", out[i].numpy())
#    cv2.waitKey()
    cv2.imwrite("generated/visualiser/CF-" + str(i) + ".png", 255*out[i].numpy())
#model_trim = keras.models.Model(input=[])