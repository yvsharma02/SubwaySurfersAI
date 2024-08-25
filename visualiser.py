from recorder import ScreenRecorder
from input_manager import InputManager
from action import Action
import os
import numpy as np
import keras
import configs
import matplotlib.pyplot as plt
from keras import layers
import settings
import config_manager

from typing import Callable
from PIL import Image
import cv2
import tensorflow as tf
import textwrap

def im_loader(path):
    raw = tf.io.read_file(path)
    tensor = tf.io.decode_image(raw)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    tensor = tf.reshape(tensor=tensor, shape=(170, 82, 3))
    return tensor

def get_layer(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return layer
        if isinstance(layer, layers.TimeDistributed):
            if (layer.layer.name == layername):
                return layer.layer
    return None

def vis_CNN3(output_dir, im_list, im_no, out_layer_count):
    model = tf.keras.models.load_model(os.path.join(settings.get_model_train_out_dir("3C95"), "model.keras"))

    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                          outputs=model.layers[out_layer_count].output)
    print(im_list)
    images = tf.convert_to_tensor([[im_loader(x) for x in im_list]])
    print(images.shape)
    res = intermediate_layer_model.predict(images)
    # Shape is (1, 3, 52, 22, 50) At this point.
    res = tf.squeeze(res, axis = 0)
    # Shape is (3, 52, 22, 50)
    res = tf.unstack(res, axis=-1)
    # Shape is [(3, 52, 22)]
    filter_no = 0
    for filter_out in res:
        images = tf.unstack(filter_out, axis=0)

        dir_path = "{}/{}/".format(output_dir, filter_no)
        if (not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        
        path = "{}/{}.png".format(dir_path, im_no)
        cv2.imwrite(path, 255*images[len(images) - 1].numpy())
        filter_no += 1


def visualise(seq_len, data_dir, func, output_dir, out_layer_count):
    dir_content = ["{}.png".format(i + 1) for i in range(0, len(os.listdir(data_dir)) - 1)]
    
    for i in range(seq_len - 1, len(dir_content) - seq_len + 1):
        images = [os.path.join(data_dir, dir_content[j]) for j in range(i - seq_len + 1, i + 1)]
        func(output_dir, images, i, out_layer_count)

def combine_color_channels(red_folder, green_folder, blue_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    n = len(os.listdir(red_folder))
    
    for i in range(1, n + 1):
        red_image_path = os.path.join(red_folder, f"{i}.png")
        green_image_path = os.path.join(green_folder, f"{i}.png")
        blue_image_path = os.path.join(blue_folder, f"{i}.png")

        if os.path.exists(red_image_path) and os.path.exists(green_image_path) and os.path.exists(blue_image_path):
            red_channel = cv2.imread(red_image_path, cv2.IMREAD_GRAYSCALE)
            green_channel = cv2.imread(green_image_path, cv2.IMREAD_GRAYSCALE)
            blue_channel = cv2.imread(blue_image_path, cv2.IMREAD_GRAYSCALE)

            rgb_image = cv2.merge((blue_channel, green_channel, red_channel))

            output_image_path = os.path.join(output_folder, f"{i}.png")
            cv2.imwrite(output_image_path, rgb_image)
        else:
            continue

def make_movie(in_folder, out_path):
    image_files = sorted([f for f in os.listdir(in_folder) if f.endswith('.png')], 
                         key=lambda x: int(x.split('.')[0]))
    # Check if output folder exists, create it if it doesn't
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if not image_files:
        print(f"No PNG images found in {in_folder}")
        return

    first_image = cv2.imread(os.path.join(in_folder, image_files[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(out_path, fourcc, 8, (width, height), isColor=True)

    for image in image_files:
        img_path = os.path.join(in_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


def find_brightest_channels(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    
    channel_brightness = []

    for subfolder in subfolders:
        image_files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
        if image_files:
            image_path = os.path.join(subfolder, image_files[0])
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            brightness = np.mean(img)
            channel_brightness.append((subfolder, brightness))

    sorted_channels = sorted(channel_brightness, key=lambda x: x[1], reverse=True)

    return [channel[0] for channel in sorted_channels[:3]]
def process_brightest_channels(root_folder, output_folder):
    brightest_channels = find_brightest_channels(root_folder)

    os.makedirs(output_folder, exist_ok=True)

    combo_name = "_".join([os.path.basename(folder) for folder in brightest_channels])
    combo_output_folder = os.path.join(output_folder, combo_name)
    os.makedirs(combo_output_folder, exist_ok=True)

    # Flip the red and green channels
    red_channel, green_channel, blue_channel = brightest_channels
    flipped_channels = [green_channel, red_channel, blue_channel]

    # Use the existing combine_color_channels function with flipped channels
    combine_color_channels(*flipped_channels, combo_output_folder)

    print(f"Processed brightest channels with flipped red and green, images saved in {combo_output_folder}")
    return combo_output_folder


config_manager.load_configs()
#visualise(3, 'generated/visualiser/loose-3C95-layer-1', vis_CNN3, 'generated/output/3C95/player/loose-3C95/2024-08-22-20-22-22', 1)
#make_movie('generated/output/cnn3-4/player/medium-cnn3-4/2024-08-22-0-45-45', 'generated/visualiser/cnn3-4/player/medium-cnn3-4/2024-08-22-0-45-45/movie.mp4')
#visualise(3, 'generated/visualiser/cnn3-4/player/medium-cnn3-4/2024-08-22-0-45-45/', vis_CNN3, 'generated/output/cnn3-4/player/medium-cnn3-4/2024-08-22-0-45-45', 1)
# process_brightest_channels('generated/visualiser/cnn3-4/player/medium-cnn3-4/2024-08-22-0-45-45/', 'generated/visualiser/cnn3-4/player/medium-cnn3-4/2024-08-22-0-45-45/movie-brightest.mp4')

import cv2
import numpy as np
from typing import List

def parse_pred(dataset_dir):
    import re
    
    pred_file_path = os.path.join(dataset_dir, 'pred.txt')
    pred_map = {}

    with open(pred_file_path, 'r') as file:
        content = file.read()
    
    # Use regex to find all matches
    pattern = r'Count\[(\d+)\]:\n(\[\[.*?]])'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for count, pred_values in matches:
        count = int(count)
        # Convert string representation to actual list of floats
        pred_values = [[float(x) for x in pred_values.strip('[]').split()]]
        pred_map[count] = pred_values

    return pred_map
    

def animate(input_folders: List[str], output_folder: str, labels: List[str]):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse predictions from the first input folder
    pred_map = parse_pred(input_folders[0])
    
    # Get list of image files that exist in all input folders
    image_files = []
    for file in os.listdir(input_folders[0]):
        if file.endswith('.png') and all(os.path.exists(os.path.join(folder, file)) for folder in input_folders):
            image_files.append(file)
    image_files.sort(key=lambda x: int(x[:-4]))  # Sort based on integer value of filename without '.png'
    
    if not image_files:
        print("No common images found across all input folders.")
        return
    
    # Process the first image to get dimensions
    first_images = [cv2.imread(os.path.join(folder, image_files[0])) for folder in input_folders]
    max_height = max(img.shape[0] for img in first_images)
    max_width = max(img.shape[1] for img in first_images)
    
    # Calculate dimensions for the combined image
    num_images = len(input_folders)
    padding = 50  # Padding on each side
    pred_width = 200  # Width for prediction column
    nn_width = 100  # Reduced width for neural network visualization
    combined_width = max_width * num_images + 50 * (num_images - 1) + 2 * padding + pred_width + nn_width  # Extra 50 pixels for each arrow, plus padding, prediction width, and neural network width
    combined_height = max_height + 300  # Increased height to accommodate text, title, and new LSTM text
    
    # Create arrow image
    arrow = np.zeros((max_height, 50, 3), dtype=np.uint8)
    cv2.arrowedLine(arrow, (10, max_height//2), (40, max_height//2), (255, 255, 255), 2, tipLength=0.3)
    
    action_labels = ['Up', 'Down', 'Left', 'Right', 'Nothing']
    
    for idx, image_file in enumerate(image_files):
        frames = []
        for folder in input_folders:
            image_path = os.path.join(folder, image_file)
            frame = cv2.imread(image_path)
            frames.append(frame)
        
        # Combine frames with fade-in effect
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        x_offset = padding  # Start after left padding
        for i, frame in enumerate(frames):
            # Calculate fade factor (0 to 1) with delay for each column, except the first
            if i == 0:
                fade_factor = 1  # No fade for the first column
            else:
                fade_factor = min(max(idx - i * 5, 0) / 3.335, 1)  # Fade-in effect over 3.335 frames for other images
            
            # Center the frame without rescaling
            y_start = (max_height - frame.shape[0]) // 2
            x_start = (max_width - frame.shape[1]) // 2
            
            # Create a blank canvas of max_height x max_width
            centered_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            
            # Place the frame in the center of the canvas
            centered_frame[y_start:y_start+frame.shape[0], x_start:x_start+frame.shape[1]] = frame
            
            # Apply fade-in effect (except for the first column)
            if i == 0:
                faded_frame = centered_frame
            else:
                faded_frame = (centered_frame * fade_factor).astype(np.uint8)
            
            # Place the frame in the combined image
            y_offset = (combined_height - max_height - 210) // 2 + 50  # Adjusted for title, text, and new LSTM text
            combined_frame[y_offset:y_offset+max_height, x_offset:x_offset+max_width] = faded_frame
            
            # Add label below the image (centered and smaller) with fade-in effect
            label = labels[i]
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x = x_offset + (max_width - label_size[0]) // 2
            label_color = (int(255 * fade_factor), int(255 * fade_factor), int(255 * fade_factor))
            cv2.putText(combined_frame, label, (label_x, y_offset + max_height + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)
            
            if i == 1:
                text = "3/16 channels"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x = x_offset + (max_width - text_size[0]) // 2
                text_color = (int(255 * fade_factor), int(255 * fade_factor), int(255 * fade_factor))
                cv2.putText(combined_frame, text, (text_x, y_offset + max_height + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)
            elif i == 2:
                text = "3/50 channels"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x = x_offset + (max_width - text_size[0]) // 2
                text_color = (int(255 * fade_factor), int(255 * fade_factor), int(255 * fade_factor))
                cv2.putText(combined_frame, text, (text_x, y_offset + max_height + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)
            elif i == 3:
                text = "3/100 channels"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x = x_offset + (max_width - text_size[0]) // 2
                text_color = (int(255 * fade_factor), int(255 * fade_factor), int(255 * fade_factor))
                cv2.putText(combined_frame, text, (text_x, y_offset + max_height + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)
            
            x_offset += max_width
            if i < num_images - 1:
                # Apply fade-in effect to the arrow
                faded_arrow = (arrow * fade_factor).astype(np.uint8)
                combined_frame[y_offset:y_offset+max_height, x_offset:x_offset+50] = faded_arrow
                x_offset += 50
        
        # Calculate fade factor for lines, text, and neural network
        nn_fade_factor = min(max(idx - (num_images) * 5, 0) / 5, 1)  # Fade-in effect over 5 frames, starting right after the last image
        line_fade_factor = min(max(idx - (num_images) * 5 - 5, 0) / 5, 1)  # Fade-in effect over 5 frames, starting 5 frames after the neural network

        # Add "Fully Connected" text with fade-in effect
        text = "Fully Connected"
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        label_x = x_offset + (max_width - label_size[0]) + 125 // 2
        cv2.putText(combined_frame, text, (label_x, y_offset + max_height + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(255 * nn_fade_factor), int(255 * nn_fade_factor), int(255 * nn_fade_factor)), 1, cv2.LINE_AA)

        # Add horizontal curly bracket and LSTM text with fade-in effect
        bracket_start = padding
        bracket_end = bracket_start + (max_width + 50) * (num_images) - 50
        bracket_y = y_offset + max_height + 100
        cv2.line(combined_frame, (bracket_start, bracket_y), (bracket_end, bracket_y), (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1)
        cv2.line(combined_frame, (bracket_start, bracket_y), (bracket_start, bracket_y - 10), (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1)
        cv2.line(combined_frame, (bracket_end, bracket_y), (bracket_end, bracket_y - 10), (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1)
        
        lstm_text = "X3 Time Distributed Input images processed parallely before LSTM"
        lstm_text_size = cv2.getTextSize(lstm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        lstm_text_x = (bracket_start + bracket_end - lstm_text_size[0]) // 2
        cv2.putText(combined_frame, lstm_text, (lstm_text_x, bracket_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1, cv2.LINE_AA)

        
        # Add arrow between last image column and neural network visualization with fade-in effect
        nn_arrow_x = x_offset
        arrow_fade_factor = nn_fade_factor  # Use the same fade factor as the neural network
        cv2.arrowedLine(combined_frame, (nn_arrow_x + 10, y_offset + max_height//2), (nn_arrow_x + 40, y_offset + max_height//2), 
                        (int(255 * arrow_fade_factor), int(255 * arrow_fade_factor), int(255 * arrow_fade_factor)), 2, tipLength=0.3)
        
        # Add neural network visualization with fade-in effect
        nn_x = nn_arrow_x + 50
        nn_y = y_offset
        nn_height = max_height
        
        # Align bracket with the one for LSTM
        nn_bracket_start = nn_x
        nn_bracket_end = nn_x + nn_width
        nn_bracket_y = bracket_y  # Use the same y-coordinate as the LSTM bracket
        cv2.line(combined_frame, (nn_bracket_start, nn_bracket_y), (nn_bracket_end, nn_bracket_y), (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1)
        cv2.line(combined_frame, (nn_bracket_start, nn_bracket_y), (nn_bracket_start, nn_bracket_y - 10), (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1)
        cv2.line(combined_frame, (nn_bracket_end, nn_bracket_y), (nn_bracket_end, nn_bracket_y - 10), (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1)

        other_end_text = "Highly Simplified"
        other_end_text_size = cv2.getTextSize(other_end_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        other_end_text_x = nn_x + (nn_width - other_end_text_size[0]) // 2
        # Align text with the LSTM text
        cv2.putText(combined_frame, other_end_text, (other_end_text_x, bracket_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(255 * line_fade_factor), int(255 * line_fade_factor), int(255 * line_fade_factor)), 1, cv2.LINE_AA)


        # Add arrow between neural network visualization and prediction column with fade-in effect
        pred_arrow_x = nn_x + nn_width
#        cv2.arrowedLine(combined_frame, (pred_arrow_x + 10, y_offset + max_height//2), (pred_arrow_x + 40, y_offset + max_height//2), 
#                        (int(255 * arrow_fade_factor), int(255 * arrow_fade_factor), int(255 * arrow_fade_factor)), 2, tipLength=0.3)

        # Add 'Flatten->LSTM->Fully Connected' text below Neural Network with fade-in effect
        nn_text = "Flatten->LSTM->Output"
        nn_text_size = cv2.getTextSize(nn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        nn_text_x = nn_x + (nn_width - nn_text_size[0]) // 2
        cv2.putText(combined_frame, nn_text, (nn_text_x, nn_y + nn_height + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(255 * nn_fade_factor), int(255 * nn_fade_factor), int(255 * nn_fade_factor)), 1, cv2.LINE_AA)
        # Add predictions to the right of the neural network visualization
        key = int(image_file.replace('.png', ''))
        predictions = pred_map[key][0] if key in pred_map else [0] * 5
        draw_neural_network(combined_frame, nn_x, nn_y, nn_width, nn_height, nn_fade_factor, predictions)
        pred_x = combined_width - pred_width + 10
        pred_y_start = y_offset  # Adjusted to align with images
        pred_spacing = nn_height // 5  # Align with neural network output nodes
        
        # Calculate fade factor for predictions (same as neural network)
        pred_fade_factor = nn_fade_factor
        
        for j, (pred, action_label) in enumerate(zip(predictions[:5], action_labels)):
            pred_y = pred_y_start + j * pred_spacing + pred_spacing // 2  # Center vertically with NN output nodes
            
            # Add action label with fade-in effect
            label_color = [int(c * pred_fade_factor) for c in (255, 255, 255)]
            cv2.putText(combined_frame, action_label, (pred_x, pred_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1, cv2.LINE_AA)
            
            # Add colored text for predicted value with fade-in effect
            pred_text = f"{pred:.2f}"
            color = (0, int(pred * 255 * pred_fade_factor), int((1 - pred) * 255 * pred_fade_factor))  # Red (0) to Green (1) in BGR with fade
            cv2.putText(combined_frame, pred_text, (pred_x + 100, pred_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        
        # Add title "What the CNN sees:" on top and center of the output image
        title = "What the CNN sees:"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        title_x = (combined_width - title_size[0]) // 2
        cv2.putText(combined_frame, title, (title_x, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save the combined frame
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, combined_frame)
    print(f"Combined images with fade-in effects, labels, colored predictions, and neural network visualization saved in {output_folder}")

def draw_neural_network(image, x, y, width, height, fade_factor, pred_list):
    # Define the number of nodes in each layer
    layers = [5, 4, 5]  # Input layer, hidden layer, output layer
    
    # Calculate the vertical spacing between nodes
    spacing = height // (max(layers) + 1)
    
    # Generate random activations for this frame
    activations = [[np.random.choice([True, False]) for _ in range(layer_size)] for layer_size in layers[0:2]]
    activations.append([True if pred_list[i] > 0.5 else False for i in range(layers[-1])])
    # Draw the nodes and connections
    for i, layer_size in enumerate(layers):
        layer_x = x + (i * width) // (len(layers) - 1)
        for j in range(layer_size):
            node_y = y + ((j + 1) * height) // (layer_size + 1)
            
            # Set color based on activation and apply fade-in effect
            color = (0, int(255 * fade_factor), 0) if activations[i][j] else (int(255 * fade_factor), int(255 * fade_factor), int(255 * fade_factor))
            
            cv2.circle(image, (layer_x, node_y), 5, color, -1)
            
            # Draw connections to the next layer
            if i < len(layers) - 1:
                next_layer_x = x + ((i + 1) * width) // (len(layers) - 1)
                for k in range(layers[i + 1]):
                    next_node_y = y + ((k + 1) * height) // (layers[i + 1] + 1)
                    if np.random.random() < 0.2:  # Only color about 20% of lines
                        green_shade = np.random.randint(50, 256)
                        line_color = (0, int(green_shade * fade_factor), 0)
                    else:
                        line_color = (int(50 * fade_factor), int(50 * fade_factor), int(50 * fade_factor))  # Light gray for other lines
                    cv2.line(image, (layer_x, node_y), (next_layer_x, next_node_y), line_color, 1)
# # Example usage:
input_videos = [
    'generated/output/3C90/player/medium-3C90/2024-08-25-18-24-24/',
    'generated/visualiser/cnn3-4/player/medium-cnn3-4/2024-08-22-0-45-45/movie-brightest.mp4/12_5_4',
]
output_video = 'generated/visualiser/cnn3-4/player/medium-cnn3-4/final'
# combine_images_with_arrows(input_videos, output_video)

dataset = 'generated/output/3C90/player/medium-3C90/2024-08-25-18-24-24'
vis_dir = dataset.replace("output", "visualiser")

# if not os.path.exists(f'{vis_dir}/layer0'):
#     visualise(3, dataset, vis_CNN3, f'{vis_dir}/layer0', 0)
# if not os.path.exists(f'{vis_dir}/layer2'):
#     visualise(3, dataset, vis_CNN3, f'{vis_dir}/layer2', 2)
# if not os.path.exists(f'{vis_dir}/layer4'):
#     visualise(3, dataset, vis_CNN3, f'{vis_dir}/layer4', 4)

# f0 = process_brightest_channels(f'{vis_dir}/layer0', f'{vis_dir}/layer0-brightest')
# f2 = process_brightest_channels(f'{vis_dir}/layer2', f'{vis_dir}/layer2-brightest')
# f4 = process_brightest_channels(f'{vis_dir}/layer4', f'{vis_dir}/layer4-brightest')

f0 = f"{vis_dir}/layer0-brightest/5_12_15"
f2 = f"{vis_dir}/layer2-brightest/1_15_19"
f4 = f"{vis_dir}/layer4-brightest/92_95_74"
# # # Example usage:
input_videos = [
    dataset,
    f0,
    f2,
    f4,
]
output_video = f'{vis_dir}/final'
animate(input_videos, output_video, ['Original', 'Conv #1', 'Conv #2', 'Conv #3'])