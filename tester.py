import tensorflow as tf
import matplotlib.pyplot as plt

import dataset_definition
import common
import os
import numpy as np
import config
from PIL import Image
import global_config
import config_loader

def test(train_config : config.ModelConfig):
    print(train_config.validation_sets)
    validation_sets : list[dataset_definition.DatasetDefinition] = [config_loader.get_dataset(key) for key in train_config.validation_sets]

    model = tf.keras.models.load_model(os.path.join(global_config.get_model_train_out_dir(train_config.model_name), "model.keras"))

    for valid in validation_sets:
        valid.load_path_label_pairs(train_config.input_image_dimension[0], train_config.input_image_dimension[1])
        print(valid.path_label_pair)
        test_dataset = dataset_definition.to_tf_dataset(valid, train_config.sequence_length
        , train_config.input_image_dimension + [3]).batch(1)

        out_dir = global_config.get_model_test_result_dir(train_config.model_name, valid.dataset_name)

        if (not os.path.exists(out_dir)):
            os.makedirs(out_dir)

        confusion = []
        for i in range(0, 5):
            confusion.append([0] * 5)

        mislabels = 0
        count = 0

        log = open(os.path.join(out_dir, "log.txt"), "w")

        for im, label in test_dataset:
            pred = model(im)
            res = np.argmax(pred).item()
            label = label.numpy().item()
            confusion[label][res] += 1

            if (res != label):
                im_arr = (np.squeeze(im[0, :, :, :, :].numpy()) * 255).astype(np.uint8)
                images = list(im_arr)

                c = 0
                for single_image in images:
                    Image.fromarray(single_image).save(os.path.join(out_dir, "{}-{}.png".format(count, c)))
                    c += 1
            
                info = "Failed in image: [{}-{}.png]: [{}] expected vs [{}] found.\n".format(count, c - 1, str(common.Action(label)), str(common.Action(res)))
                print(info)
                log.write(info)
                mislabels += 1

            count += 1


        log.write("\n\n___________CONFSUSION___________\n\n")
        for r in confusion:
            print(r)
            log.write("{}\n".format(r))

        print("Accuracy: {}".format(float(count - mislabels) / float(count)))
        log.write("Accuracy: {}".format(float(count - mislabels) / float(count)))

        log.close()