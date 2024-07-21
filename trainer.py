import keras
from keras import models, layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt

from common import CustomDataSet
import common
import datetime
import os

custom_train_dataset = CustomDataSet("data/2024-07-18-23-35-35", (397, 859))
custom_test_dataset = CustomDataSet("data/2024-07-18-23-35-35", (397, 859))

model = models.Sequential()
model.add(layers.Input(shape=(397, 859, 3)))
model.add(layers.Conv2D(32, (9, 9), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5))

model.summary()
model.compile(optimizer='adam',
              loss= losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

training = custom_train_dataset.get_dataset().batch(8)
testing = custom_test_dataset.get_dataset().batch(8)

history = model.fit(training, epochs = 1, verbose = 1, validation_data=testing)

out_dir = os.path.join("out/", common.date_to_dirname(datetime.datetime.now()))

if (not os.path.exists(out_dir)):
    os.mkdir(out_dir)

model.save("model.keras")

model.save(os.path.join(out_dir, "model.keras"))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()