import config
import os
from PIL import Image
import shutil

for dataset in os.listdir(config.ORIGINAL_DATA_DIR):
    save_dir = os.path.join(config.DOWNSCALED_DATA_DIR, dataset)
    read_dir = os.path.join(config.ORIGINAL_DATA_DIR, dataset)


    if (os.path.exists(save_dir)):
        continue
    
    os.mkdir(save_dir)
    print ("Generating: " + save_dir)
    
#    files = os.listdir(os.path.join(config.ORIGINAL_DATA_DIR, dataset))
#    files.sort()
#    files = [str(i) + '.png' for i in range(1, len(files))]
#    files = files[:-(config.END_TRIM_COUT + 1)]
#    print(files)
#    files.append("commands.txt")
    
    with (open(os.path.join(read_dir, "commands.txt"), "r") as commands_txt):
        lines = commands_txt.read().splitlines()[:-(config.END_TRIM_COUT)]
        if (len(lines) > config.BATCH_SIZE):
            lines = lines[0:config.MAX_BATCH_SIZE]

        print(len(lines))
        
        with (open(os.path.join(save_dir, "commands.txt"), "w") as out):
            for line in lines:
                out.write(line + "\n")
        
        for line in lines:
            im, action, date, nl = line.split(';')
            im = im.rstrip().lstrip() + ".png"

            og_path = os.path.join(read_dir, im)
            save_path = os.path.join(save_dir, im)

            Image.open(og_path).resize(tuple(reversed(config.INPUT_IMAGE_DIMENSIONS))).save(save_path)