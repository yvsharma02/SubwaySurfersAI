import os
import settings
from keras import models, layers, losses

def get_architecture() -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=(5, 172, 80, 3)))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(96, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
#    model.add(layers.TimeDistributed(layers.Dropout(0.15)))
    model.add(
        layers.TimeDistributed(layers.Dense(200, activation='relu'))
    )
    model.add(layers.Dropout(0.175))
    model.add(
        layers.TimeDistributed(layers.Dense(150, activation='relu'))
    )
    model.add(layers.Dropout(0.175))
    model.add(
        layers.LSTM(100, activation='tanh', return_sequences=False)
    )
    model.add(
        layers.Dense(50, activation='relu')
    )
    model.add(
        layers.Dense(25, activation='relu')
    )

    model.add(layers.Dense(5))
    model.add(layers.Softmax())

    model.summary()
    model.compile(optimizer='adam',
                loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def generate_dataset_list(mirrors = True, end_trim = 50):
    dirs = os.listdir(settings.ORIGINAL_DATA_DIR)

    format = "[ \"{}\", [0, {}], [], {} ],"

    with open(settings.DATASET_JSON_OUT_FILE, "w") as file:
        for dir in dirs:
            file.write(format.format(dir, end_trim, "true" if mirrors else "false"))

def architecture_json(name : str):
    with open(os.path.join(settings.ARCH_JSON_OUT_DIR, name + ".json"), "w") as file:
       file.write(get_architecture().to_json())