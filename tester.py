import keras
from keras import models, layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt

from common import CustomDataSet
import common
import datetime
import os
import numpy as np
import config
from PIL import Image

test_dataset_custom = CustomDataSet(os.path.join(config.DOWNSCALED_DATA_DIR, config.VALIDATION_DATASET))

test_dataset = test_dataset_custom.get_dataset(nothing_skip_rate=config.TEST_DATA_NOTHING_SKIP_RATE).batch(1)
model = tf.keras.models.load_model(os.path.join(config.MODEL_OUTPUT_DIR, config.PLAY_MODEL, "model.keras"))

out_dir = os.path.join(config.TESTER_OUTPUT_DIR, config.PLAY_MODEL, common.date_to_dirname(datetime.datetime.now()))

if (not os.path.exists(out_dir)):
    os.makedirs(out_dir)

confusion = []
for i in range(0, 5):
    confusion.append([0] * 5)

mislabels = 0
count = 0

log = open(os.path.join(out_dir, "log.txt"), "w")

for im, label in test_dataset:
    pred = model(im)
    res = np.argmax(pred).item()
    label = label.numpy().item()
    confusion[label][res] += 1

    if (res != label):
        im_arr = (np.squeeze(im[0, :, :, :, :].numpy()) * 255).astype(np.uint8)
        images = list(im_arr)

        c = 0
        for single_image in images:
            Image.fromarray(single_image).save(os.path.join(out_dir, "{}-{}.png".format(count, c)))
            c += 1
    
        info = "Failed in image: [{}-{}.png]: [{}] expected vs [{}] found.\n".format(count, c - 1, str(common.Action(label)), str(common.Action(res)))
        print(info)
        log.write(info)
        mislabels += 1

    count += 1


log.write("\n\n___________CONFSUSION___________\n\n")
for r in confusion:
    print(r)
    log.write("{}\n".format(r))

print("Accuracy: {}".format(float(count - mislabels) / float(count)))
log.write("Accuracy: {}".format(float(count - mislabels) / float(count)))

log.close()