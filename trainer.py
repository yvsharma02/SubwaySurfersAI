from keras import models, callbacks
import tensorflow as tf
import matplotlib.pyplot as plt

import dataset_definition
import os

import config_loader
import config
import global_config

def train(train_config : config.ModelConfig):
    out_dir = global_config.get_model_train_out_dir(train_config.model_name)

    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    class CallbackHandler(callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            self.model.save(os.path.join(out_dir, "model_epoch_{}.keras".format(epoch)))

    print("Final Input Shape: {}".format(train_config.get_final_input_shape()))
    model = config_loader.get_model_generator(train_config.model_architecture_name)(train_config.get_final_input_shape())

    train_set = config_loader.get_dataset(train_config.train_set_name)
    train_set.load_path_label_pairs(train_config.input_image_dimension[0], train_config.input_image_dimension[1])

    data = dataset_definition.to_tf_dataset(train_set, train_config.sequence_length, train_config.input_image_dimension + [3])
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(batch_size=train_config.batch_size)
    test_batch_count = int(len(data) * train_config.testing_fraction)

    testing = data.take(test_batch_count)
    training = data.skip(test_batch_count)
    print(testing)
    print(training)

    history = model.fit(training, epochs = train_config.epoch, verbose = 1, validation_data=testing, callbacks=[CallbackHandler()])

    # if (not os.path.exists(out_dir)):
    #     os.mkdir(out_dir)

    print(history.__dict__)
    model.save(os.path.join(out_dir, "model.keras"))

    with open(os.path.join(out_dir, "summary.txt"), "w") as file:
        file.write("Accuracy: " + str(history.history['accuracy']))
        file.write("\n")
        file.write("Validation Accuracy: " + str(history.history['val_accuracy']))
        file.write("\nTraining Dataset Size: " + str(len(training)))
        file.write("\nValidation Dataset Size: " + str(len(testing)))
        file.write("\n__________________________\n")
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(out_dir, 'history.png'))

    return model