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

#custom_train_dataset = CustomDataSet("data/2024-07-21-14-6-6", IM_DIM)
test_dataset_custom = CustomDataSet(os.path.join(config.DOWNSCALED_DATA_DIR, config.TEST_DATASET))

test_dataset = test_dataset_custom.get_dataset().batch(1)
model = tf.keras.models.load_model(os.path.join(config.MODEL_OUTPUT_DIR, config.PLAY_MODEL, "model.keras"))

for im, label in test_dataset:
    pred = model(im)
    print(str(np.argmax(pred)) + " vs " + str(label))

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# plt.show()