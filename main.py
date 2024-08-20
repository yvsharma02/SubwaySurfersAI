import config_manager
import trainer
import tester
import os

import settings

config_manager.load_configs()

for key in config_manager.get_all_train_models():
    if (not os.path.exists(os.path.join(settings.get_model_train_out_dir(key), "model.keras"))):
        print("Training Model: [{}]".format(key))
        model = trainer.train(config_manager.get_model_config(key))
    else:
        print("[{}] Already trained. Skipping.".format(key))

    print("Testing Model: [{}]".format(key))
    tester.test(config_manager.get_model_config(key))