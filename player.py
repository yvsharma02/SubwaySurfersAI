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
import cv2

from PIL import Image

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
im_save_count = 0
lastFrameTime = datetime.datetime.now()

out_dir = os.path.join(config.PLAYER_OUTPUT_DIR, config.PLAY_MODEL, common.date_to_dirname(datetime.datetime.now()))

if (not os.path.exists(out_dir)):
    os.makedirs(out_dir)

pred_file = open(os.path.join(out_dir, "pred.txt"), "w")

def update(count, last_action_time):
    global context_window_len, lastFrameTime, im_save_count
    if (last_action_time == None):
        last_action_time = datetime.datetime.now()
    try:
        arr = recorder.capture()
        im = Image.fromarray(arr).resize(tuple(reversed(config.INPUT_IMAGE_DIMENSIONS)))
        im_save_count += 1
        im.save(os.path.join(out_dir, str(im_save_count) + ".png"))

        tensor = np.asarray(im, dtype=np.float32).reshape(config.TRAINING_IMAGE_DIMENSIONS)
        tensor = tensor / 255

        context_window.append(tensor)
        context_window_len += 1

        current_time = datetime.datetime.now()
        time_diff = (current_time - lastFrameTime).total_seconds()
        fps = 1.0 / time_diff
        lastFrameTime = current_time
        print("FPS: {}".format(fps))

        if (context_window_len > config.SEQUENCE_LEN):
            context_window_len -= 1
            context_window.remove(context_window[0])

        if(context_window_len < config.SEQUENCE_LEN):
            return

        dataset = tf.data.Dataset.from_tensors(tf.stack(context_window)).batch(1)

        pred = model.predict(dataset)

        ranks = np.argsort(pred[0])
        diff = pred[0][ranks[-1]] - pred[0][ranks[-2]]

        perform = False
        predicted_action = Action(np.argmax(pred[0]))

        first_choice = ranks[-1]
        second_choice = ranks[-2]

        predicted_action = Action(first_choice)

        if (first_choice != int(Action.SWIPE_LEFT) and first_choice != int(Action.SWIPE_RIGHT)):
            perform = True
        else:
            if (pred[0][first_choice] + pred[0][second_choice] >= config.MIN_ACTION_CONFIDENCE[first_choice]):
                perform = True

        # for i in range(0, len(ranks)):
        #     action_ind = ranks[-(i + 1)]
        #     confidence = pred[0][action_ind]

        #     combined_confidence = confidence

        #     if (action_ind == int(Action.SWIPE_LEFT)):
        #         if (ranks[-(i + 2)] == int(Action.SWIPE_RIGHT)):
        #             combined_confidence += pred[0][-(i + 2)]
        #     elif (action_ind == int(Action.SWIPE_RIGHT)):
        #         if (ranks[-(i + 2)] == int(Action.SWIPE_LEFT)):
        #             combined_confidence += pred[0][-(i + 2)]

        #     if (combined_confidence > config.MIN_ACTION_CONFIDENCE[action_ind]):
        #         perform = True
        #         break

        performed = False
        if (predicted_action != Action.DO_NOTHING and perform and (datetime.datetime.now() - last_action_time).total_seconds() >= config.PLAYER_ACTION_PERFORM_COOLDOWN):
            performed = True
            input_manager.perform_action(predicted_action)

        info = "Count[{}]:\n{}\n[Max is: {} @ {}]\n[Frame Time: {}]\nPerformed Status: {}\n________________________________________\n\n".format(im_save_count, str(pred), str(Action(ranks[-1])), pred[0][ranks[-1]], time_diff, performed)
        print(info)
        pred_file.write(info)

        if (performed):
            return datetime.datetime.now()
        return last_action_time
#            time.sleep(config.PLAYER_ACTION_PERFORM_COOLDOWN)

        
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

pred_file.close()
input_manager.flush()