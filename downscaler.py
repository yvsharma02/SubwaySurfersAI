import config
import os
from PIL import Image
import shutil

for dataset in os.listdir(config.ORIGINAL_DATA_DIR):
    save_dir = os.path.join(config.DOWNSCALED_DATA_DIR, dataset)
    
    if (os.path.exists(save_dir)):
        continue
    
    os.mkdir(save_dir)
    
    for file in os.listdir(os.path.join(config.ORIGINAL_DATA_DIR, dataset)):
        og_path = os.path.join(config.ORIGINAL_DATA_DIR, dataset, file)
    
        save_path = os.path.join(save_dir, file)

        if (file.endswith(".png")):
            Image.open(og_path).resize(config.INPUT_IMAGE_DIMENSIONS).save(save_path)
        else:
            shutil.copy(og_path, save_path)
