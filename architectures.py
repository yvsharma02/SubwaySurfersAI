from keras import models, layers, losses

# Had to revert to declaring model in python because tf 1 does not support saving compiled models (NOTE: Confirm this.)

def get_model(final_input_shape) -> models.Model:
    model = models.Sequential()
#    model.add(layers.Input(shape=(5, 172, 80, 3)))
    model.add(layers.Input(shape=final_input_shape))
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

architecture_generator_map = {
    'test_arch': get_model
}
