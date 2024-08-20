import configs
import os
from PIL import Image
from action import Action
import shutil
import settings
import custom_dataset

def downscale(dataset_dir, height : int, width : int, mirror = True, replace = False):
    downscaled_dir = os.path.join(settings.DOWNSCALED_DIR, "{}x{}".format(height, width), dataset_dir)
    og_dir = os.path.join(settings.ORIGINAL_DATA_DIR, dataset_dir)
    if (os.path.exists(downscaled_dir) and not replace):
        return

    if (not os.path.exists(downscaled_dir)):
        os.makedirs(downscaled_dir)

        with (open(os.path.join(og_dir, "commands.txt"), "r") as commands_txt):
            txt = commands_txt.read()
            lines = txt.splitlines()

            with (open(os.path.join(downscaled_dir, "commands.txt"), "w") as out):
                for line in lines:
                    out.write(line + "\n")
            
            for line in lines:
                im, action, date, nl = line.split(';')
                im = im.rstrip().lstrip() + ".png"

                og_path = os.path.join(og_dir, im)
                save_path = os.path.join(downscaled_dir, im)

                Image.open(og_path).resize((width, height)).save(save_path)

            #Mirrored
            if (mirror):
                txt = txt.replace(" " + str(int(Action.SWIPE_LEFT)) + "; ", "*")
                txt = txt.replace(" " + str(int(Action.SWIPE_RIGHT)) + "; ", " " + str(int(Action.SWIPE_LEFT)) + "; ")
                txt = txt.replace("*", " " + str(int(Action.SWIPE_RIGHT)) + "; ")
                save_dir = os.path.join(settings.DOWNSCALED_DIR, "{}x{}".format(height, width), dataset_dir + " - reversed")
                os.mkdir(save_dir)
                lines = txt.splitlines()

                with (open(os.path.join(save_dir, "commands.txt"), "w") as out):
                    for line in lines:
                        out.write(line + "\n")
                
                for line in lines:
                    im, action, date, nl = line.split(';')
                    im = im.rstrip().lstrip() + ".png"

                    og_path = os.path.join(og_dir, im)
                    save_path = os.path.join(save_dir, im)

                    Image.open(og_path).resize((width, height)).transpose(Image.FLIP_LEFT_RIGHT).save(save_path)