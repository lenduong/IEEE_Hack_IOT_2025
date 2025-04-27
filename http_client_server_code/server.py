# #----------- Imports for HTTP Flask Server -----------#
# from flask import Flask
# from flask import jsonify
# from flask import request
# from flask import render_template
# #-----------------------------------------------------#

# #-------------- Miscellaneous Imports ----------------#
# import json
# # import subprocess
# #-----------------------------------------------------#

# # ----------------- GLobal Commands ------------------#
# app = Flask('RaspberryPi Mailbox Server') 
# #-----------------------------------------------------#

# # --------------------------- Code for HTTP Server ----------------------------#
# # Set up home page for web front end
# # Use http://localhost:8080/ to access front end
# # Change based on whether command is ON or OFF
# @app.route('/')
    
# # Custom callback function to receive image from RPi
# @app.route('/send_image', methods=['POST'])
# def post_image_callback():
#     # Ensure the content type is JPEG
#     if request.content_type != 'image/jpeg':
#             return jsonify({"error": "Invalid content type. Only JPEG is supported."}), 400

#     # Save the Image
#     image_data = request.data
#     save_path = f'uploads/test_image.jpg'
#     with open(save_path, "wb") as f:
#             f.write(image_data)
#     # subprocess.Popen(["python3", "../MiDaS-master/run.py"])

#     # Return a response to client to confirm the request
#     return jsonify({"message1": f"Image received and saved as {save_path}"}), 200


# # --------------------------------- Main ----------------------------------------#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=2020) # Begin the server
# # -------------------------------------------------------------------------------#


    
#----------- Imports for HTTP Flask Server -----------#
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
#-----------------------------------------------------#

#--------- Imports for loading the ML model ----------#
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
import matplotlib.pyplot as plt
#-----------------------------------------------------#

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
    save_path = f'uploads/test_image.jpg'
    with open(save_path, "wb") as f:
            f.write(image_data)

    # Call deploy() to predict the image
    global LED_command
    LED_command = deploy()

    # Return a response to client to confirm the request, along with the LED command
    return jsonify({"message1": f"Image received and saved as {save_path}", "message2":LED_command}), 200
# -------------------------------------------------------------------------------#


# --------------------------- Code for ML Prediction ----------------------------#
# Define image pre-processing in a function -----------------
def image_preprocessor(image_path):
  img = Image.open(image_path).convert('L')
  img = img.resize(size=(128, 128))
  img_array = (np.array(img) > 100)*255  # Convert to a numpy array
  plt.imshow(img_array)

  img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
  img_array = img_array.reshape(1, 128, 128, 1)  # Reshape for model input
                # (batch_size, image dimension, single channel grayscale)
  return img_array

# Feed the image into the trained model to predict -----------
def deploy():
    image_path = '/mnt/c/Users/leduo/Desktop/EE250-Collab/final_project/http/uploads/test_image.jpg'
    img_array = image_preprocessor(image_path)
    prediction = loaded_model(img_array) # use a direct call for small input size
    predict_value = np.argmax(prediction)

    print("Model Prediction: ")
    print(predict_value)

    # Taking model output and converting it to be ON or OFF for LEDs
    # Even values produce LED ON
    if predict_value % 2 == 0:
        print("LED ON")
        return True
    else: # Odd value produce LED OFF
        print("LED OFF")
        return False
# -------------------------------------------------------------------------------#


# --------------------------------- Main ----------------------------------------#
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) # Begin the server
# -------------------------------------------------------------------------------#


    
