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

import tensorflow as tf
#from keras import models, layers, losses


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

run_start_time = datetime.datetime.now()
recorder = ScreenRecorder(config.SCREEN_NAME)
input_manager = InputManager(recorder)

model = tf.keras.models.load_model('out/2024-07-22-2-1-1/model.keras')

def update(count, last_action_time):
    if (last_action_time == None):
        last_action_time = datetime.datetime.now()
    try:
        im = recorder.capture() 
        tensor = np.asarray(im, dtype=np.float32)
        tensor = tensor / 255
        tensor = tensor.reshape(config.FINAL_IMAGE_SHAPE)
        dataset = tf.data.Dataset.from_tensors(tensor).batch(1)

        pred = model.predict(dataset)
#        print(Action(np.argmax(pred[0])))

        ranks = np.argsort(pred[0])
        diff = pred[0][ranks[-1]] - pred[0][ranks[-2]]

        if (diff < .5): # Indecisive
            print("Indecisive")
            return last_action_time

        print(pred[0])
        input_manager.perform_action(Action(np.argmax(pred[0])))
#        time.sleep(.333333)
    except:
        pass

    return last_action_time

frame_counter = 0
last_fps_marker_time = datetime.datetime.now()
last_fps_marker_frame = 0

print("Press any key to start")
keyboard.read_key()

c = 0
last_action_time = None
while True:
    new_action_time = update(c, last_action_time)
    if (new_action_time != last_action_time):
        c += 1
    last_action_time = new_action_time
    frame_counter += 1

    if ((datetime.datetime.now() - last_fps_marker_time).seconds >= 1):
#        print("FPS: " + str(frame_counter - last_fps_marker_frame))
        last_fps_marker_time = datetime.datetime.now()
        last_fps_marker_frame = frame_counter

    if keyboard.is_pressed('q'):
        break

recorder.flush()