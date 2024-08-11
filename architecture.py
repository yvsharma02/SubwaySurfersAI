from keras import models, layers, losses
import config

def get_model() -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=config.TRAINNIG_DATA_DIMENSIONS))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.Dropout(0.30))
    
    model.add(
        layers.Dense(125, activation='tanh')
    )
    model.add(layers.Dropout(0.25))
    model.add(
        layers.LSTM(40, activation='relu', return_sequences=False)
    )
    model.add(
        layers.Dense(25, activation='tanh')
    )
    model.add(layers.Dense(5))
    model.add(layers.Softmax())

    model.summary()
    model.compile(optimizer='adam',
                loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model