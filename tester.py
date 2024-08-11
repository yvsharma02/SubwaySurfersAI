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

# model = tf.keras.models.load_model(os.path.join(config.MODEL_OUTPUT_DIR, config.PLAY_MODEL, "model.keras"))
# layer_names = [layer.name for layer in model.layers]
# layer_outputs = [layer.output for layer in model.layers]

# feature_map_model = tf.keras.models.Model(input=model.input, output=layer_outputs)
# image_path= "generated/downscaled/2024-07-26-23-2-2"
# img = load_img(image_path, target_size=(150, 150))  
# input = img_to_array(img)                           
# input = x.reshape((1,) + x.shape)                   
# input /= 255.0


#custom_train_dataset = CustomDataSet("data/2024-07-21-14-6-6", IM_DIM)
test_dataset_custom = CustomDataSet(os.path.join(config.DOWNSCALED_DATA_DIR, config.TEST_DATASET))

test_dataset = test_dataset_custom.get_dataset(only_nothing_skip_rate=0.975).batch(1)
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
#    print(pred)
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