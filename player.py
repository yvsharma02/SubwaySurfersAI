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
import PIL
import cv2

import tensorflow as tf
#from keras import models, layers, losses


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

run_start_time = datetime.datetime.now()
recorder = ScreenRecorder(config.SCREEN_NAME)
input_manager = InputManager(recorder)

model = tf.keras.models.load_model(os.path.join(config.MODEL_OUTPUT_DIR, config.PLAY_MODEL, "model.keras"))

#last_frame_pred = Action.DO_NOTHING
#held_pred_for = 0

context_window_len = 0
context_window = []

lastFrameTime = datetime.datetime.now()

def update(count, last_action_time):
    global context_window_len, lastFrameTime
    if (last_action_time == None):
        last_action_time = datetime.datetime.now()
    try:
        im = cv2.resize(recorder.capture(), tuple(reversed(config.INPUT_IMAGE_DIMENSIONS)), interpolation=cv2.INTER_CUBIC)
#        im = recorder.capture().resize(), PIL.Image.NEAREST)
        tensor = np.asarray(im, dtype=np.float32).reshape(config.TRAINING_IMAGE_DIMENSIONS)
        tensor = tensor / 255

        context_window.append(tensor)
        context_window_len += 1

        current_time = datetime.datetime.now()
        diff = (current_time - lastFrameTime).total_seconds()
        fps = 1.0 / diff
        lastFrameTime = current_time
        print("FPS: {}".format(fps))

        if (context_window_len > config.SEQUENCE_LEN):
            context_window_len -= 1
            context_window.remove(context_window[0])

        if(context_window_len < config.SEQUENCE_LEN):
            return

#        print(tensor.shape)
        dataset = tf.data.Dataset.from_tensors(tf.stack(context_window)).batch(1)

        pred = model.predict(dataset)
#        print(Action(np.argmax(pred[0])))

        ranks = np.argsort(pred[0])
        diff = pred[0][ranks[-1]] - pred[0][ranks[-2]]

        print("{} [max is {} @ {}]".format(pred, str(Action(ranks[-1])), pred[0][ranks[-1]]))
        if (diff < config.MIN_PLAYER_CONFIDENCE): # Indecisive
            if (Action(ranks[-2]) == Action.DO_NOTHING):
                print("Indecisive")
                return last_action_time

        predicted_action = Action(np.argmax(pred[0]))
        if (predicted_action != Action.DO_NOTHING):
            input_manager.perform_action(predicted_action)
            time.sleep(.125)

        
    except Exception as e:
        print("Something has gone totally wrong: " + str(e))

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