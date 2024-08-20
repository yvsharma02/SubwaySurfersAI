import os

DATA_GEN_ACTION_PERFORM_COOLDOWN = .25
STD_OUT_FOR_SUBPROCESS = "stdout.txt"
ORIGINAL_DATA_DIR = "generated/data/original"

RECORD_FPS = 8

CAPTURE_LTRB_OFFSET = (9, 38, -9, -9)
SCREEN_NAME = "Android Emulator - Pixel_4a_API_33:5554"

ARCHITECTURES_CONFIG_DIR = "configs/architectures"
PLAYERS_CONFIG_DIR = "configs/players"
DATASETS_CONFIG_DIR = "configs/datasets"
MODELS_CONFIG_DIR = "configs/models"

ORIGINAL_DATA_DIR = "generated/data/original"
DOWNSCALED_DIR = "generated/data/downscaled"

DATASET_JSON_OUT_FILE = "generated/utils/datasets.json"
ARCH_JSON_OUT_DIR = "generated/utils/"

resultant_models_root = "generated/output/"

def get_model_train_out_dir(model_name : str):
    return os.path.join(resultant_models_root, model_name)

def get_model_test_result_dir(model_name : str, train_config_name : str):
    return os.path.join(get_model_train_out_dir(model_name), "evaluation", train_config_name)

def get_model_player_result_dir(model_name : str, player_name : str):
    return os.path.join(get_model_train_out_dir(model_name), "player", player_name)

def date_to_dirname(run_start_time):
    return str(run_start_time.date()) + "-" + str(run_start_time.hour) + "-" + str(run_start_time.minute) + "-" + str(run_start_time.minute)
