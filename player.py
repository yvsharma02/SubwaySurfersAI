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

last_frame_pred = Action.DO_NOTHING
held_pred_for = 0

def update(count, last_action_time):
    if (last_action_time == None):
        last_action_time = datetime.datetime.now()
    try:
        im = recorder.capture().resize(tuple(reversed(config.INPUT_IMAGE_DIMENSIONS)))
        tensor = np.asarray(im, dtype=np.float32)
        tensor = tensor / 255
        tensor = tensor.reshape(config.TRAINING_IMAGE_DIMENSIONS)
#        print(tensor.shape)
        dataset = tf.data.Dataset.from_tensors(tensor).batch(1)

        pred = model.predict(dataset)
#        print(Action(np.argmax(pred[0])))

        ranks = np.argsort(pred[0])
        diff = pred[0][ranks[-1]] - pred[0][ranks[-2]]

        # if (diff < .8): # Indecisive
        #     print("Indecisive")
        #     return last_action_time

        predicted_action = Action(np.argmax(pred[0]))
#        print(pred[0])
        global last_frame_pred
        global held_pred_for
        if (predicted_action != Action.DO_NOTHING):
            if (last_frame_pred == predicted_action):
                held_pred_for += 1
                if (predicted_action != Action.SWIPE_UP and held_pred_for >= config.ACTION_HOLD_FRAME_COUT) or (held_pred_for >= config.ACTION_HOLD_FRAME_COUT + config.UP_EXTRA_HOLD_TIME):
                    print(str(predicted_action) + " held for " + str(held_pred_for))
                    input_manager.perform_action(predicted_action)
                    last_frame_pred = Action.DO_NOTHING
                    held_pred_for = 0
                    time.sleep(.15)
            else:
                last_frame_pred = predicted_action
                held_pred_for = 0

#            im.show()

        #TODO: Only perform action if it is consistent for the last 3-4 frames.    
        
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