from tensorflow import keras
from keras import models, layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt

from common import CustomDataSet

train_set = CustomDataSet("data/2024-07-18-23-35-35", (397, 859))
test_set = CustomDataSet("data/2024-07-18-23-40-40", (397, 859))


model = models.Sequential()
model.add(layers.Input(shape=(397, 859, 3)))
model.add(layers.Conv2D(32, (9, 9), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((4, 4)))
#model.add(layers.Conv2D(64, (4, 4), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5))

model.summary()
model.compile(optimizer='adam',
              loss= losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

training_set = train_set.get_dataset().batch(1)

#print(training_set.batch(1))
#test_images, test_labels = train_set.get_dataset()

history = model.fit(training_set, epochs = 5, verbose = 1)

plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

#test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)