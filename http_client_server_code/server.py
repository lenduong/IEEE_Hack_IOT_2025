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
#-----------------------------------------------------#


# --------------------------- Code for HTTP Server ----------------------------#
# Set up home page for web front end
# Use http://localhost:8080/ to access front end
# Change based on whether command is ON or OFF
@app.route('/')
    
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

    # Return a response to client to confirm the request
    return jsonify({"message1": f"Image received and saved as {save_path}"}), 200


# --------------------------------- Main ----------------------------------------#
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) # Begin the server
# -------------------------------------------------------------------------------#


    
