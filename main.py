import config_loader
import trainer
import tester

config_loader.load_configs()

for key in config_loader.get_all_train_models():
    print("Training Model: {}".format(key))
    model = trainer.train(config_loader.get_model_config(key))

    print("Testing Model: {}".format(key))
    tester.test(model)