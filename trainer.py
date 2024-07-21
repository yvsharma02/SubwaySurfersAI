import keras
from keras import models, layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt

from common import CustomDataSet
import common
import datetime
import os

import config

testing_dir = "data/2024-07-22-0-10-10"
training_dir = [os.path.join("data/", d) for d in os.listdir("data/")]
training_dir.remove(testing_dir)

custom_train_dataset = CustomDataSet(training_dir[0])
custom_train_dataset = common.combine_custom_datasets([CustomDataSet(td) for td in training_dir])
custom_test_dataset = CustomDataSet(testing_dir)

custom_train_dataset.summary()

model = models.Sequential()
model.add(layers.Input(shape=config.FINAL_IMAGE_SHAPE))
model.add(layers.Conv2D(16, (9, 9), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(16, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten())
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(5))
model.add(layers.Softmax())

model.summary()
model.compile(optimizer='adam',
              loss= losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

training = custom_train_dataset.get_dataset().batch(config.BATCH_SIZE)
testing = custom_test_dataset.get_dataset().batch(config.BATCH_SIZE)

history = model.fit(training, epochs = config.EPOCH, verbose = 1, validation_data=testing)

out_dir = os.path.join("out/", common.date_to_dirname(datetime.datetime.now()))

if (not os.path.exists(out_dir)):
    os.mkdir(out_dir)

model.save(os.path.join(out_dir, "model.keras"))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()