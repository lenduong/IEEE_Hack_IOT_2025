"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import time

import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

#----------- Imports for HTTP Flask Server -----------#
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
#-----------------------------------------------------#

# #--------- Imports for loading the ML model ----------#
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras
# from PIL import Image
# import matplotlib.pyplot as plt
# #-----------------------------------------------------#

#-------------- Miscellaneous Imports ----------------#
from PIL import Image
import argparse
import json
from datetime import datetime
from threading import Lock
#-----------------------------------------------------#

# ----------------- GLobal Commands ------------------#
app = Flask('RaspberryPi Mailbox Server') 
loaded_model = keras.models.load_model('handNums_model-1104.h5') # load the trained ML model
LED_command = False # global flag variable for LED
#-----------------------------------------------------#

# --------------------------- Code for HTTP Server ----------------------------#
# Set up home page for web front end
# Use http://localhost:8080/ to access front end
# Change based on whether command is ON or OFF
@app.route('/')
def home():
    global LED_command
    if LED_command == True:
        return render_template('frontEnd-ON.html')
    return render_template('frontEnd-OFF.html')
    
# Custom callback function to receive image from RPi
@app.route('/send_image', methods=['POST'])
def post_image_callback():
    # Ensure the content type is JPEG
    if request.content_type != 'image/jpeg':
            return jsonify({"error": "Invalid content type. Only JPEG is supported."}), 400

    # Save the Image
    image_data = request.data
    save_path = f'input/test_image.jpg'
    with open(save_path, "wb") as f:
            f.write(image_data)

    # Call deploy() to predict the image
    global buzzer_command
    buzzer_command = command

    # Return a response to client to confirm the request, along with the LED command
    return jsonify({"message1": f"Image received and saved as {save_path}", "message2":buzzer_command}), 200
# -------------------------------------------------------------------------------#


first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    one_d_depth = normalized_depth
    print("1st normalized_depth min: ", one_d_depth.min(), "normalized_depth max: ", one_d_depth.max())
    print("one_d_depth shape: ", one_d_depth.shape)
    normalized_depth *= 3
    # print("2nd normalized_depth min: ", normalized_depth.min(), "normalized_depth max: ", normalized_depth.max())
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    # print("3nd normalized_depth min: ", right_side.min(), "normalized_depth max: ", right_side.max())
    

    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side, right_side
    else:
        print("--------------- normalized_depth min: ", one_d_depth.min(), "normalized_depth max: ", one_d_depth.max())
        print("one_d_depth shape: ", one_d_depth.shape)
        return one_d_depth, np.concatenate((image, right_side), axis=1)


def run(input_path, output_path, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    # get input
    if input_path is not None:
        image_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(image_names)
    else:
        print("No input path specified. Grabbing images from camera.")

    # create output folder
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")
    command = 'straight'
    if input_path is not None:
        while(True):
            if output_path is None:
                print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
            for index, image_name in enumerate(image_names):

                print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

                # input
                original_image_rgb = utils.read_image(image_name)  # in [0, 1]
                image = transform({"image": original_image_rgb})["image"]

                # compute
                with torch.no_grad():
                    prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                        optimize, False)
                    

                # output
                if output_path is not None:
                    filename = os.path.join(
                        output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type
                    )
                    if not side:
                        utils.write_depth(filename, prediction, grayscale, bits=2)
                    else:
                        original_image_bgr = np.flip(original_image_rgb, 2)
                        grayscale, content = create_side_by_side(original_image_bgr*255, prediction, grayscale=True)
                        cv2.imwrite(filename + ".png", content)
                        right_flag = False
                        left_flag = False
                        middle_flag = False
                        for i in range(0,2):
                            for col in range(0 + (i*320),320 + (i*320)):
                                for row in range(0, 480):
                                    if grayscale[row][col] > 720:
                                        if i == 0:
                                            left_flag = True
                                        else :
                                            right_flag = True
                                            if left_flag and right_flag:
                                                middle_flag = True
                        
                        if middle_flag:
                            command = "middle"
                            print("Object detected in the MIDDLE! -------------------")
                        elif right_flag:
                            command = "right"
                            print("Object detected on the RIGHT! >>>>>>>>>>>>>>>>>>>>>>>>")
                        elif left_flag: 
                            command = "left"
                            print("Object detected on the LEFT! <<<<<<<<<<<<<<<<<<<<<<")
                        else: 
                            print("No object detected :( ()()()()()()()()()()()()()()")


                        # Create or load an example image (let's make a blank white image)
                        image = np.ones((500, 500, 3), dtype=np.uint8) * 255  # 500x500 white image
                        image = content/255
                        # Define your text
                        text = command

                        # Choose the position (x, y) where you want the text
                        position = (50, 250)  # 50 pixels right, 250 pixels down

                        # Set font, scale, color, and thickness
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 2
                        color = (0, 0, 0)  # Black color in BGR
                        thickness = 3

                        # Put the text on the image
                        image_with_text = cv2.putText(image.copy(), text, position, font, font_scale, color, thickness)

                        # If you want to visualize it
                        # cv2.imshow('Image with Text', image_with_text)

                        # --------------------- Grascale processing 

                    utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))


    else:
        with torch.no_grad():
            fps = 3
            video = VideoStream(0).start()
            time_start = time.time()
            frame_index = 0
            while True:
                command = 'straight'
                frame = video.read()
                if frame is not None:
                    original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
                    image = transform({"image": original_image_rgb/255})["image"]

                    prediction = process(device, model, model_type, image, (net_w, net_h),
                                         original_image_rgb.shape[1::-1], optimize, True)
                

                    original_image_bgr = np.flip(original_image_rgb, 2) if side else None
                    grayscale, content = create_side_by_side(original_image_bgr, prediction, grayscale=True)
                    np.set_printoptions(threshold=np.inf, linewidth=640, suppress=True) 
                    print("Prediction type: ", type(grayscale))
                    print("Prediction shape (rxc): ", grayscale.shape)
                    print("Prediction min: ", grayscale.min(), "Prediction max: ", grayscale.max())
                    # print(grayscale)
                    # --------------------- Grascale processing -----------------------
                    right_flag = False
                    left_flag = False
                    middle_flag = False
                    for i in range(0,2):
                        for col in range(0 + (i*320),320 + (i*320)):
                            for row in range(0, 480):
                                if grayscale[row][col] > 720:
                                    if i == 0:
                                        left_flag = True
                                    else :
                                        right_flag = True
                                        if left_flag and right_flag:
                                            middle_flag = True
                    
                    if middle_flag:
                        command = "middle"
                        print("Object detected in the MIDDLE! -------------------")
                    elif right_flag:
                        command = "right"
                        print("Object detected on the RIGHT! >>>>>>>>>>>>>>>>>>>>>>>>")
                    elif left_flag: 
                        command = "left"
                        print("Object detected on the LEFT! <<<<<<<<<<<<<<<<<<<<<<")
                    else: 
                        print("No object detected :( ()()()()()()()()()()()()()()")


                    # Create or load an example image (let's make a blank white image)
                    image = np.ones((500, 500, 3), dtype=np.uint8) * 255  # 500x500 white image
                    image = content/255
                    # Define your text
                    text = command

                    # Choose the position (x, y) where you want the text
                    position = (50, 250)  # 50 pixels right, 250 pixels down

                    # Set font, scale, color, and thickness
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2
                    color = (0, 0, 0)  # Black color in BGR
                    thickness = 3

                    # Put the text on the image
                    image_with_text = cv2.putText(image.copy(), text, position, font, font_scale, color, thickness)

                    # If you want to visualize it
                    cv2.imshow('Image with Text', image_with_text)

                    # --------------------- Grascale processing -----------------------


                    # cv2.imshow('MiDaS Depth Estimation - Press Escape to close window ', content/255)

                    if output_path is not None:
                        filename = os.path.join(output_path, 'Camera' + '-' + model_type + '_' + str(frame_index))
                        cv2.imwrite(filename + ".png", content)

                    alpha = 0.1
                    if time.time()-time_start > 0:
                        fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
                        time_start = time.time()
                    print(f"\rFPS: {round(fps,2)}", end="")

                    if cv2.waitKey(1) == 27:  # Escape key
                        break

                    frame_index += 1
        print()

    print("Finished")
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )

    args = parser.parse_args()


    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    global command = run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height, args.square, args.grayscale)
    app.run(debug=True, host='0.0.0.0', port=8080) # Begin the server

    
