from recorder import ScreenRecorder
from input_manager import InputManager
from common import Action
import datetime
import os
import numpy as np
import config

import common
import global_config
import keyboard
import time
import global_config

from PIL import Image
from keras import models

import tensorflow as tf
#from keras import models, layers, losses

def start_player(player_config : config.PlayerConfig):

    recorder = ScreenRecorder(global_config.SCREEN_NAME)
    input = InputManager(recorder)

    context_window = []
    im_save_count = 0

    out_dir = os.path.join(global_config.get_model_player_result_dir(player_config.model_name, player_config.player_name), 
            common.date_to_dirname(datetime.datetime.now()))

    model = models.models.load_model(os.path.join(global_config.get_model_test_result_dir(player_config.model_name), "model.keras"))
    pred_file = open(os.path.join(out_dir, "pred.txt"), "w")

    last_performed_action = Action.DO_NOTHING

    while (True):
        frame_start_time = datetime.datetime.now()

        if keyboard.is_pressed('q'):
            break

        arr = recorder.capture()
        im = Image.fromarray(arr).resize(reversed(player_config.get_model_config().input_image_dimension))
        im_save_count += 1
        im.save(os.path.join(out_dir, str(im_save_count) + ".png"))

        tensor = np.asarray(im, dtype=np.float32).reshape(config.TRAINING_IMAGE_DIMENSIONS)
        tensor = tensor / 255

        context_window.append(tensor)

        if (len(context_window) > player_config.get_model_config().SEQUENCE_LEN):
            context_window.remove(context_window[0])

        if(len(context_window) < player_config.get_model_config().SEQUENCE_LEN):
           continue

        if (len(context_window) > player_config.get_model_config().SEQUENCE_LEN):
            context_window.remove(context_window[0])
        
        dataset = tf.data.Dataset.from_tensors(tf.stack(context_window)).batch(1)
        pred = model.predict(dataset)

        predicted_action = Action.DO_NOTHING

        vertical_confidence = (pred[0][int(Action.SWIPE_DOWN)] + pred[0][int(Action.SWIPE_UP)])
        horizontal_confidence = (pred[0][int(Action.SWIPE_LEFT)] + pred[0][int(Action.SWIPE_RIGHT)])

        if (vertical_confidence > player_config.MIN_VERTICAL_CONFIDENCE):
            predicted_action = Action.SWIPE_UP if pred[0][int(Action.SWIPE_UP)] > pred[0][int(Action.SWIPE_DOWN)] else Action.SWIPE_DOWN
        elif (horizontal_confidence > player_config.MIN_HORIZONTAL_CONFIDENCE):
            predicted_action = Action.SWIPE_LEFT if pred[0][int(Action.SWIPE_LEFT)] > pred[0][int(Action.SWIPE_RIGHT)] else Action.SWIPE_RIGHT


        performed = False
        if (predicted_action != Action.DO_NOTHING):
            if  (last_performed_action != predicted_action or frames_since_last_action >= player_config.same_action_wait_frames):
                performed = True
                last_performed_action = predicted_action
                input.perform_action(predicted_action)
            else:
                print("Skipping due to cooldown")

        info = "Count[{}]:\n{}\n[Max is: {} @ {}]\n[Performing: {}]\n[Frame Time: {}]\nPerformed Status: {}\n________________________________________\n\n".format(im_save_count, str(pred), str(Action(ranks[-1])), pred[0][ranks[-1]], str(predicted_action), time_diff, performed)
        print(info)
        pred_file.write(info)

        if (performed):
            frames_since_last_action = 0
            return datetime.datetime.now()
        else:
            frames_since_last_action += 1

        frame_end_time = datetime.datetime.now()

        frame_time = (frame_end_time - frame_start_time).total_seconds()
        target_time = 1.0 / player_config.target_fps
        remaining_time = frame_time - target_time

        if (remaining_time > 0):
            time.sleep(remaining_time)