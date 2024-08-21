import config_manager
import trainer
import tester
import os

import settings

import tensorflow as tf
import numpy as np
import random

config_manager.load_configs()


for key in config_manager.get_all_train_models():
    
    tf.random.set_seed(settings.SEED)
    np.random.seed(settings.SEED)
    random.seed(settings.SEED)

    if (not os.path.exists(os.path.join(settings.get_model_train_out_dir(key), "model.keras"))):
        print("Training Model: [{}]".format(key))
        model = trainer.train(config_manager.get_model_config(key))
    else:
        print("[{}] Already trained. Skipping.".format(key))

    print("Testing Model: [{}]".format(key))
    tester.test(config_manager.get_model_config(key))