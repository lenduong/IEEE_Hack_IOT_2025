#--------- Import for Flask/ HTTP Functions ---------#
import requests
#----------------------------------------------------#

#----------- Imports for Camera Functions -----------#
import cv2
from PIL import Image, ImageEnhance
#----------------------------------------------------#

#--- Imports for Controlling LED and Potentimeter ---#
import RPi.GPIO as GPIO
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008
#----------------------------------------------------#

#--------------- Imports for Threading --------------#
import time
import threading
#----------------------------------------------------#

#---------------- Capture and Send the First Image ----------------------#
# This is necessary to initialize the global variable "response" to the correct type.
#     "response" is used in led_pot() function, which is on a separte thread, and need 
#     to be initialized as a global variable

# Create url to send to server (using server's IP addr)
url = "http://172.20.10.12:8080/send_image"

# Capture the image -----------------------------------
# Start camera
cam = cv2.VideoCapture(0)

# Caputure the imgage
ret, frame = cam.read()
cam.release()

# Adjust image to increase prediction accuracy --------
# Get image dimensions
height, width, _ = frame.shape

# Crop the image to the center square
size = min(height, width)
x_start = (width - size) // 2
y_start = (height - size) // 2
cropped_frame = frame[y_start:y_start+size, x_start:x_start+size]

# Resize the cropped image to 128x128
resized_frame = cv2.resize(cropped_frame, (128, 128))

# Save the cropped image
cv2.imwrite("cropped_image_128x128.jpg", resized_frame)

# Adjust contrast
alpha = 2.0  # Contrast control
beta = -50     # Brightness control
adjusted = cv2.convertScaleAbs(resized_frame, alpha=alpha, beta=beta)

# Turn the image into jpg file
_, buffer = cv2.imencode('.jpg', adjusted)

# Send the image via HTTP POST -----------------------
headers = {"Content-Type": "image/jpeg"}  # Indicate JPEG format
response = requests.post(url, data=buffer.tobytes(), headers=headers)
time.sleep(3) 

# Update flag to check for new command
new_response = True
#-----------------------------------------------------------------------------------------#


#---------------------- Function to Control LED and Potentiometer ------------------------#
def led_pot():
        # ------------------------Set up for LED and Potentiometer---------------------------
        # Using physical pin 11 to blink Red LED, 15 to blink Yellow LED
        GPIO.setmode(GPIO.BOARD)
        red_led = [11]
        yellow_led = [15]
        GPIO.setup(red_led, GPIO.OUT)
        GPIO.setup(yellow_led, GPIO.OUT)
        
        # Hardware SPI configuration:
        SPI_PORT   = 0
        SPI_DEVICE = 0
        mcp = Adafruit_MCP3008.MCP3008(spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE))
        
        # led variable tells the function which light is being controlled
        led = [15]

        while True:
            global new_response
            if new_response ==True:
                global reponse
                # ----------------------Process response from server---------------------------
                # Parse the JSON file
                # Message1 : image upload confirmation message
                # Message2 : command to turn on or off light, ouput by ML model
                message = response.json()
            
                # Check the response status code, if 200 then proceed
                if response.status_code == 200:
                    print("ooooooooooooooooooooooooooooooooooooooooooo")
                    print("Image uploaded successfully")
                    print(message["message1"])
                    print("Light On: ",message["message2"])
                    print("Potentiometer: " ,mcp.read_adc(0))
                    print("ooooooooooooooooooooooooooooooooooooooooooo")

                    # If potentiometer is turned to upper half, turn on Red LED
                    # If potentiometer is turned to lower half, turn on Yellow LED
                    if (mcp.read_adc(0) > 530):
                        led = [11] #RED
                    elif (mcp.read_adc(0) <= 500):
                        led = [15] #YELLOW

                    # If message2 = True turn on light, else turn off
                    if message["message2"] and led == [11]:
                        # Red LED
                        GPIO.output(led, GPIO.HIGH)
                        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                        print("Turning on Red LED")
                        print("Potentiometer Channel 0: ", mcp.read_adc(0))
                        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    elif message["message2"] and led == [15]: 
                        # Yellow LED
                        GPIO.output(led, GPIO.HIGH)
                        print("-------------------------------------------")
                        print("Turning on Yellow LED")
                        print("Potentiometer Channel 0: ", mcp.read_adc(0))
                        print("-------------------------------------------")
                    elif not message["message2"]:
                        GPIO.output(led, GPIO.LOW)
                else:
                    print("Error uploading image:", response.status_code)

                new_response = False
#-------------------------------------------------------------------------------------#
            

#----------------------------------- Main --------------------------------------------#
if __name__ == '__main__':
        #-------------------- Start a New Thread for led_pot() ------------------#
        thread = threading.Thread(target=led_pot) # Spawn a thread to run led_pot() in a separate thread 
        thread.daemon = True # kill the thread as soon as the main program exit
        thread.start() # start the thread executing
        #------------------------------------------------------------------------#

        #---------------- Capture and Send the First Image ----------------------#
        while True:
            # Create url to send to server (using server's IP addr)
            url = "http://192.168.91.71:8080/send_image"
            
            # Capture the image -----------------------------------
            # Start camera
            cam = cv2.VideoCapture(0)
            
            # Caputure the imgage
            ret, frame = cam.read()
            cam.release()
            
            # Adjust image to increase prediction accuracy --------
            # Get image dimensions
            height, width, _ = frame.shape
            
            # Crop the image to the center square
            size = min(height, width)
            x_start = (width - size) // 2
            y_start = (height - size) // 2
            cropped_frame = frame[y_start:y_start+size, x_start:x_start+size]
            
            # Resize the cropped image to 128x128
            resized_frame = cv2.resize(cropped_frame, (128, 128))
            
            # Save the cropped image
            cv2.imwrite("cropped_image_128x128.jpg", resized_frame)
            
            # Adjust contrast
            alpha = 2.0  # Contrast control
            beta = -50     # Brightness control
            adjusted = cv2.convertScaleAbs(resized_frame, alpha=alpha, beta=beta)
            
            # Turn the image into jpg file
            _, buffer = cv2.imencode('.jpg', adjusted)
            
            # Send the image via HTTP POST -----------------------
            headers = {"Content-Type": "image/jpeg"}  # Indicate JPEG format
            response = requests.post(url, data=buffer.tobytes(), headers=headers)
            time.sleep(3) 
            
            # Update flag to check for new command
            new_response = True
        #-----------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

        
def buzzer():
        # Set the GPIO pin number
        buzzer_pin = 23

        # Set the GPIO pin as output
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(buzzer_pin, GPIO.OUT)
        while True:
                GPIO.output(buzzer_pin, GPIO.HIGH) # Turn buzzer on
                time.sleep(1)
                GPIO.output(buzzer_pin, GPIO.LOW) # Turn buzzer off
                time.sleep(1)

 #-------------------- Start a New Thread for led_pot() ------------------#
        thread = threading.Thread(target=buzzer) # Spawn a thread to run led_pot() in a separate thread 
        thread.daemon = True # kill the thread as soon as the main program exit
        thread.start() # start the thread executing
        #------------------------------------------------------------------------#
import RPi.GPIO as GPIO
import time

