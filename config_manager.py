import json
import os
import settings
import configs
import custom_dataset
import architectures

def load_json(path : str):
    with (open(path, "r") as file):
        return json.load(file)

def save_json(path: str, obj):
    with (open(path, "w") as file):
        file.write(json.dumps(obj.__dict__))

models_config = {}
datasets_config = {}
players_config = {}

def load_configs():

    global models_config, datasets_config, players_config

    for tc in os.listdir(settings.MODELS_CONFIG_DIR):
        name = tc.removesuffix('.json')
        models_config[name] = configs.ModelConfig()
        models_config[name].__dict__.update(load_json(os.path.join(settings.MODELS_CONFIG_DIR, tc)))
        models_config[name].model_name = name

    for dc in os.listdir(settings.DATASETS_CONFIG_DIR):
        name = dc.removesuffix('.json')
        datasets_config[name] = custom_dataset.CustomDataset()
        datasets_config[name].__dict__.update(load_json(os.path.join(settings.DATASETS_CONFIG_DIR, dc)))
        datasets_config[name].dataset_name = name

    for pc in os.listdir(settings.PLAYERS_CONFIG_DIR):
        name = pc.removesuffix('.json')
        players_config[name] = configs.PlayerConfig()
        players_config[name].__dict__.update(load_json(os.path.join(settings.PLAYERS_CONFIG_DIR, pc)))
        players_config[name].player_name = name


def get_model_config(name : str) -> configs.ModelConfig:
    return models_config[name]

def get_dataset(name : str) -> custom_dataset.CustomDataset:
    return datasets_config[name]

def get_player_config(name : str) -> configs.PlayerConfig:
    return players_config[name]

def get_model_generator(name : str) -> str:
    return architectures.architecture_generator_map[name]

def get_all_train_models():
    return models_config.keys()

def get_all_datasets():
    return datasets_config.keys()