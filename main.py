import config_loader
import trainer
import tester
import os

import global_config

config_loader.load_configs()

for key in config_loader.get_all_train_models():
    if (os.path.exists(os.path.join(global_config.get_model_train_out_dir(key), "model.keras"))):
        print("{} already exists. Skipping".format(key))
    print("Training Model: {}".format(key))
    model = trainer.train(config_loader.get_model_config(key))

    print("Testing Model: {}".format(key))
    tester.test(config_loader.get_model_config(key))