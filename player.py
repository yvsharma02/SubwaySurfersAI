from recorder import ScreenRecorder
from input_manager import InputManager
from action import Action
import datetime
import os
import numpy as np
import configs

import action
import settings
import keyboard
import time
import settings

from PIL import Image
from keras import models
import config_manager

import tensorflow as tf

def start_player(player_config : configs.PlayerConfig):
    model_config : configs.ModelConfig = config_manager.get_model_config(player_config.model_name)

    recorder = ScreenRecorder(settings.SCREEN_NAME)
    input = InputManager(recorder)

    context_window = []
    im_save_count = 0

    out_dir = os.path.join(settings.get_model_player_result_dir(player_config.model_name, player_config.player_name), 
            action.date_to_dirname(datetime.datetime.now()))

    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    model = models.load_model(os.path.join(settings.get_model_train_out_dir(player_config.model_name), "model.keras"))
    pred_file = open(os.path.join(out_dir, "pred.txt"), "w")

    last_performed_action = Action.DO_NOTHING
    frames_since_last_action = 0

    print("Press f to start")
    while (not keyboard.is_pressed('f')):
        time.sleep(0.1)

    while (True):
        frame_start_time = datetime.datetime.now()

        if keyboard.is_pressed('q'):
            break
        
        downscaled_shape = (model_config.input_image_dimension[1], model_config.input_image_dimension[0])
        arr = recorder.capture()
        if (arr is None):
            print("Capture failed")
            continue
        im = Image.fromarray(arr).resize(downscaled_shape)
        im_save_count += 1
        im.save(os.path.join(out_dir, str(im_save_count) + ".png"))
#        print(im)
        tensor = np.asarray(im, dtype=np.float32).reshape((int(model_config.input_image_dimension[0]), int(model_config.input_image_dimension[1]), 3))
        tensor = tensor / 255

        context_window.append(tensor)

        if (len(context_window) > model_config.sequence_length):
            context_window.remove(context_window[0])

        if(len(context_window) < model_config.sequence_length):
           continue

        if (len(context_window) > model_config.sequence_length):
            context_window.remove(context_window[0])
        
        dataset = tf.data.Dataset.from_tensors(tf.stack(context_window)).batch(1)
        pred = model.predict(dataset)

        predicted_action = Action.DO_NOTHING

        vertical_confidence = (pred[0][int(Action.SWIPE_DOWN)] + pred[0][int(Action.SWIPE_UP)])
        horizontal_confidence = (pred[0][int(Action.SWIPE_LEFT)] + pred[0][int(Action.SWIPE_RIGHT)])

        if (vertical_confidence > player_config.min_vertical_confidence):
            predicted_action = Action.SWIPE_UP if pred[0][int(Action.SWIPE_UP)] > pred[0][int(Action.SWIPE_DOWN)] else Action.SWIPE_DOWN
        elif (horizontal_confidence > player_config.min_horizontal_confidence):
            predicted_action = Action.SWIPE_LEFT if pred[0][int(Action.SWIPE_LEFT)] > pred[0][int(Action.SWIPE_RIGHT)] else Action.SWIPE_RIGHT

        ranks = np.argsort(pred[0])

        if (predicted_action == Action.DO_NOTHING):
            if (pred[0][int(Action.DO_NOTHING)] < player_config.min_nothing_confidence):
                predicted_action = Action(ranks[-2])

        performed = False
        if (predicted_action != Action.DO_NOTHING):
            if  (last_performed_action != predicted_action or frames_since_last_action >= player_config.same_action_wait_frames):
                performed = True
                last_performed_action = predicted_action
                input.perform_action(predicted_action)
            else:
                print("Skipping due to cooldown")

        if (performed):
            frames_since_last_action = 0
        else:
            frames_since_last_action += 1

        frame_end_time = datetime.datetime.now()

        frame_time = (frame_end_time - frame_start_time).total_seconds()
        target_time = 1.0 / player_config.target_fps
        remaining_time = target_time - frame_time

        
        info = "Count[{}]:\n{}\n[Max is: {} @ {}]\n[Performing: {}]\n[Frame Time: {}]\nPerformed Status: {}\n________________________________________\n\n".format(im_save_count, str(pred), str(Action(ranks[-1])), pred[0][ranks[-1]], str(predicted_action), frame_time, performed)
        print(info)
        pred_file.write(info)

        if (remaining_time > 0.2):
           print(remaining_time)
           time.sleep(remaining_time)

config_manager.load_configs()
start_player(config_manager.get_player_config("loose-3C70"))