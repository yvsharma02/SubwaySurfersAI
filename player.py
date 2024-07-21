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

import tensorflow as tf
#from keras import models, layers, losses

run_start_time = datetime.datetime.now()
recorder = ScreenRecorder(config.SCREEN_NAME)
input_manager = InputManager(recorder)

#def get_time_path(run_start_time):
#    return str(run_start_time.date()) + "-" + str(run_start_time.hour) + "-" + str(run_start_time.minute) + "-" + str(run_start_time.minute)

def get_data_dir(run_start_time):
    res = "data/" + common.date_to_dirname(run_start_time) + "/"
    if (not os.path.exists(res)):
        os.makedirs(res)

    return res


model = tf.keras.models.load_model('out/2024-07-21-1-5-5/model.keras')

def update(count, last_action_time):
    if (last_action_time == None):
        last_action_time = datetime.datetime.now()
    
    im = recorder.capture() 
    tensor = np.asarray(im, dtype=np.float32)
    tensor = tensor / 255
#    print("________")
#    print(tensor.shape)
    tensor = tensor.reshape(IM_SHAPE + tuple([3]))
    dataset = tf.data.Dataset.from_tensors(tensor).batch(1)

    for data in dataset:
        pred = model(data)
        if (Action(np.argmax(pred)) != Action.DO_NOTHING):
            print(Action(np.argmax(pred)))

    return last_action_time

frame_counter = 0
last_fps_marker_time = datetime.datetime.now()
last_fps_marker_frame = 0

c = 0
last_action_time = None
while True:
    try:
        new_action_time = update(c, last_action_time)
    except:
        pass
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