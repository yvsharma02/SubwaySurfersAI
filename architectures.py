from keras import models, layers, losses


def CNN3_50DROP(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(50, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(100, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dropout(0.5)))
    model.add(
        layers.TimeDistributed(layers.Dense(125, activation='relu'))
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.TimeDistributed(layers.Dense(100, activation='relu'))
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.LSTM(100, activation='tanh', return_sequences=False)
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(50, activation='relu')
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(25, activation='relu')
    )
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5))
    model.add(layers.Softmax())

    model.summary()
    model.compile(optimizer='adam',
                loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def CNN3Named(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu', name="conv1")))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3), name='pool-1')))
    model.add(layers.TimeDistributed(layers.Conv2D(50, (5, 5), activation='relu', name="conv2")))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2), name='pool-2')))
    model.add(layers.TimeDistributed(layers.Conv2D(100, (3, 3), activation='relu', name="conv3")))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2), name='pool-3')))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dropout(0.4)))
    model.add(
        layers.TimeDistributed(layers.Dense(125, activation='relu'))
    )
    model.add(layers.Dropout(0.4))
    model.add(
        layers.TimeDistributed(layers.Dense(100, activation='relu'))
    )
    model.add(layers.Dropout(0.4))
    model.add(
        layers.LSTM(100, activation='tanh', return_sequences=False)
    )
    model.add(layers.Dropout(0.3))
    model.add(
        layers.Dense(50, activation='relu')
    )
    model.add(layers.Dropout(0.2))
    model.add(
        layers.Dense(25, activation='relu')
    )
    model.add(layers.Dropout(0.125))

    model.add(layers.Dense(5))
    model.add(layers.Softmax())

    model.summary()
    model.compile(optimizer='adam',
                loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def CNN4(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (2, 2), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dropout(0.4)))
    model.add(
        layers.TimeDistributed(layers.Dense(125, activation='relu'))
    )
    model.add(layers.Dropout(0.4))
    model.add(
        layers.TimeDistributed(layers.Dense(100, activation='relu'))
    )
    model.add(layers.Dropout(0.4))
    model.add(
        layers.LSTM(100, activation='tanh', return_sequences=False)
    )
    model.add(layers.Dropout(0.3))
    model.add(
        layers.Dense(50, activation='relu')
    )
    model.add(layers.Dropout(0.2))
    model.add(
        layers.Dense(25, activation='relu')
    )
    model.add(layers.Dropout(0.125))

    model.add(layers.Dense(5))
    model.add(layers.Softmax())

    model.summary()
    model.compile(optimizer='adam',
                loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

architecture_generator_map = {
    '3C': CNN3Named,
    '3C_50DROP': CNN3_50DROP,
    '4C': CNN4
}
