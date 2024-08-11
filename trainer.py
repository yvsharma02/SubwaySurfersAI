import keras
from keras import models, layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt

from common import CustomDataSet, Action
import common
import datetime
import os
import shutil

import config

training_dir = [os.path.join(config.DOWNSCALED_DATA_DIR, d) for d in os.listdir(config.DOWNSCALED_DATA_DIR)]
testing_dir = os.path.join(config.DOWNSCALED_DATA_DIR, config.TEST_DATASET)
training_dir.remove(testing_dir)

custom_train_dataset = common.combine_custom_datasets([CustomDataSet(td) for td in training_dir])

testing_dataset = CustomDataSet(testing_dir)

model = models.Sequential()
model.add(layers.Input(shape=config.TRAINNIG_DATA_DIMENSIONS))
model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
model.add(layers.TimeDistributed(layers.Flatten()))

model.add(
    layers.LSTM(1024, activation='relu', return_sequences=False)
)
model.add(
    layers.Dense(150, activation='relu')
)
model.add(layers.Dense(5))
model.add(layers.Softmax())

model.summary()
model.compile(optimizer='adam',
              loss= losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'f1'])

training = custom_train_dataset.get_dataset()
testing = testing_dataset.get_dataset()
training = training.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(batch_size=config.BATCH_SIZE)

history = model.fit(training, epochs = config.EPOCH, verbose = 1, validation_data=testing)

out_dir = os.path.join(config.MODEL_OUTPUT_DIR, common.date_to_dirname(datetime.datetime.now()))

if (not os.path.exists(out_dir)):
    os.mkdir(out_dir)

model.save(os.path.join(out_dir, "model.keras"))

with open(os.path.join(out_dir, "summary.txt"), "w") as file:
    file.write("Accuracy: " + str(history.history['accuracy']))
    file.write("\n")
    file.write("Validation Accuracy: " + str(history.history['val_accuracy']))
    file.write("\nTraining Dataset Size: " + str(len(training) * config.BATCH_SIZE))
    file.write("\nValidation Dataset Size: " + str(len(testing) * config.BATCH_SIZE))
    file.write("\n__________________________\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))

shutil.copy("config.py", os.path.join(out_dir, "config.py"))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.savefig(os.path.join(out_dir, 'history.png'))