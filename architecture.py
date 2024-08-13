from keras import models, layers, losses
import config

def get_model() -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=config.TRAINNIG_DATA_DIMENSIONS))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dropout(0.15)))
    
    model.add(
        layers.TimeDistributed(layers.Dense(300, activation='relu'))
    )
    model.add(layers.Dropout(0.15))
    model.add(
        layers.LSTM(50, activation='relu', return_sequences=False)
    )
    model.add(
        layers.Dense(50, activation='relu')
    )
    model.add(
        layers.Dense(50, activation='tanh')
    )
    model.add(layers.Dense(5))
    model.add(layers.Softmax())

    model.summary()
    model.compile(optimizer='adam',
                loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model