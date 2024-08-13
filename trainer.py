import keras
from keras import models, layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt

from common import CustomDataSet, Action
import common
import datetime
import os
import shutil
import architecture

import config

training_dir = [os.path.join(config.DOWNSCALED_DATA_DIR, d) for d in os.listdir(config.DOWNSCALED_DATA_DIR)]
validation_dir = os.path.join(config.DOWNSCALED_DATA_DIR, config.VALIDATION_DATASET)
training_dir.remove(validation_dir)
training_dir.remove(validation_dir + ' - reversed')

custom_train_dataset = common.combine_custom_datasets([CustomDataSet(td) for td in training_dir])

model = architecture.get_model()

data = custom_train_dataset.get_dataset(nothing_skip_rate=config.TRAIN_DATA_NOTHING_SKIP_RATE)
data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(batch_size=config.BATCH_SIZE)
test_batch_count = int(len(data) * config.TRAINING_FRACTION)

testing = data.take(test_batch_count)
training = data.skip(test_batch_count)

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
shutil.copy("architecture.py", os.path.join(out_dir, "architecture.py"))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.savefig(os.path.join(out_dir, 'history.png'))