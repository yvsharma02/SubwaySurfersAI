from keras import models, layers, losses

# Had to revert to declaring model in python because tf 1 does not support saving compiled models (NOTE: Confirm this.)

def cnn31(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (5, 5), activation='relu')))
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

def cnn32(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(96, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
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

def cnn41(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(96, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(144, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
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

def cnn33(final_input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(96, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
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

def cnn21(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Flatten()))
#    model.add(layers.TimeDistributed(layers.Dropout(0.15)))
    model.add(
        layers.TimeDistributed(layers.Dense(125, activation='relu'))
    )
    model.add(layers.Dropout(0.175))
    model.add(
        layers.TimeDistributed(layers.Dense(100, activation='relu'))
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

def cnn22(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(
        layers.TimeDistributed(layers.Dense(125, activation='relu'))
    )
    model.add(layers.Dropout(0.175))
    model.add(
        layers.TimeDistributed(layers.Dense(100, activation='relu'))
    )
    model.add(layers.Dropout(0.175))

    model.add(
        layers.Dense(50, activation='relu')
    )
    model.add(
        layers.LSTM(25, activation='tanh', return_sequences=False)
    )
    model.add(
        layers.Dense(25, activation='relu')
    )
    model.add(
        layers.Dense(15, activation='relu')
    )

    model.add(layers.Dense(5))
    model.add(layers.Softmax())

    model.summary()
    model.compile(optimizer='adam',
                loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def cnn23(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
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

def cnn34(final_input_shape) -> models.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=final_input_shape))
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((3, 3))))
    model.add(layers.TimeDistributed(layers.Conv2D(50, (5, 5), activation='relu')))
    model.add(layers.TimeDistributed(layers.AveragePooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(100, (3, 3), activation='relu')))
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
    'cnn3-1': cnn31,
    'cnn3-2': cnn32,
    'cnn3-3': cnn33,
    'cnn4-1': cnn41,
    'cnn2-1': cnn21,
    'cnn2-2': cnn22,
    'cnn2-3': cnn23,
    'cnn3-4': cnn34
}
