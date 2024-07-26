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

confusion = []
for i in range(0, 5):
    confusion.append([0] * 5)

#print(confusion[0])

indecisive = 0

for im, label in test_dataset:
    pred = model(im)
    res = np.argmax(pred).item()
    # if (pred[0][res].numpy().item() < .9):
    #    indecisive += 1
    #    continue
    print(pred)
#    print("Classified {} as {}", res, label.numpy().item())
    confusion[label.numpy().item()][res] += 1

for r in confusion:
    print(r)
print("Indecisive: ", indecisive)

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# plt.show()