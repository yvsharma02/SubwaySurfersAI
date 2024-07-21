import keras
from keras import models, layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt

from common import CustomDataSet
import common
import datetime
import os
import numpy as np

#custom_train_dataset = CustomDataSet("data/2024-07-21-14-6-6", IM_DIM)
test_dataset_custom = CustomDataSet("data/2024-07-22-0-10-10")

test_dataset = test_dataset_custom.get_dataset().batch(1)
model = tf.keras.models.load_model('out/2024-07-22-1-54-54/model.keras')

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