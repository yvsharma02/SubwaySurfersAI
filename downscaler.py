import config
import os
from PIL import Image
from common import Action
import shutil
import global_config
import dataset_definition

def downscale(dataset_dir, height : int, width : int, mirror = True, replace = False):
    downscaled_dir = os.path.join(global_config.DOWNSCALED_DIR, "{}x{}".format(height, width), dataset_dir)
    og_dir = os.path.join(global_config.ORIGINAL_DATA_DIR, dataset_dir)
    if (os.path.exist(downscaled_dir) and not replace):
        return

    if (not os.path.exists(downscaled_dir)):
        os.makedirs(downscaled_dir)

        with (open(os.path.join(og_dir, "commands.txt"), "r") as commands_txt):
            txt = commands_txt.read()
            lines = txt.splitlines()

            with (open(os.path.join(save_dir, "commands.txt"), "w") as out):
                for line in lines:
                    out.write(line + "\n")
            
            for line in lines:
                im, action, date, nl = line.split(';')
                im = im.rstrip().lstrip() + ".png"

                og_path = os.path.join(og_dir, im)
                save_path = os.path.join(save_dir, im)

                Image.open(og_path).resize((width, height)).save(save_path)

            #Mirrored
            if (mirror):
                txt = txt.replace(" " + str(int(Action.SWIPE_LEFT)) + "; ", "*")
                txt = txt.replace(" " + str(int(Action.SWIPE_RIGHT)) + "; ", " " + str(int(Action.SWIPE_LEFT)) + "; ")
                txt = txt.replace("*", " " + str(int(Action.SWIPE_RIGHT)) + "; ")
                save_dir = os.path.join(global_config.DOWNSCALED_DIR, "{}x{}".format(height, width), dataset_dir + " - reversed")
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
        

# def downscale():

#     for dataset in os.listdir(config.ORIGINAL_DATA_DIR):
#         save_dir = os.path.join(config.DOWNSCALED_DATA_DIR, dataset)
#         read_dir = os.path.join(config.ORIGINAL_DATA_DIR, dataset)

#         if (os.path.exists(save_dir)):
#             continue
        
#         os.mkdir(save_dir)
#         print ("Generating: " + save_dir)
        
#         with (open(os.path.join(read_dir, "commands.txt"), "r") as commands_txt):
#             txt = commands_txt.read()
#             lines = txt.splitlines()[:-(config.END_TRIM_COUT)]
#             if (len(lines) > config.BATCH_SIZE):
#                 lines = lines[0:config.MAX_RUN_SIZE]

#             print(len(lines))
            
#             with (open(os.path.join(save_dir, "commands.txt"), "w") as out):
#                 for line in lines:
#                     out.write(line + "\n")
            
#             for line in lines:
#                 im, action, date, nl = line.split(';')
#                 im = im.rstrip().lstrip() + ".png"

#                 og_path = os.path.join(read_dir, im)
#                 save_path = os.path.join(save_dir, im)

#                 Image.open(og_path).resize(tuple(reversed(config.INPUT_IMAGE_DIMENSIONS))).save(save_path)

#             #Mirrored
#             txt = txt.replace(" " + str(int(Action.SWIPE_LEFT)) + "; ", "*")
#             txt = txt.replace(" " + str(int(Action.SWIPE_RIGHT)) + "; ", " " + str(int(Action.SWIPE_LEFT)) + "; ")
#             txt = txt.replace("*", " " + str(int(Action.SWIPE_RIGHT)) + "; ")
#             save_dir = os.path.join(config.DOWNSCALED_DATA_DIR, dataset + str(" - reversed"))
#             os.mkdir(save_dir)
#             lines = txt.splitlines()[:-(config.END_TRIM_COUT)]

#             with (open(os.path.join(save_dir, "commands.txt"), "w") as out):
#                 for line in lines:
#                     out.write(line + "\n")
            
#             for line in lines:
#                 im, action, date, nl = line.split(';')
#                 im = im.rstrip().lstrip() + ".png"

#                 og_path = os.path.join(read_dir, im)
#                 save_path = os.path.join(save_dir, im)

#                 Image.open(og_path).resize(tuple(reversed(config.INPUT_IMAGE_DIMENSIONS))).transpose(Image.FLIP_LEFT_RIGHT).save(save_path)