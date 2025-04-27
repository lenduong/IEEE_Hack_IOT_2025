#--------- Import for Flask/ HTTP Functions ---------#
import requests
#----------------------------------------------------#

#----------- Imports to call system command -----------#
import subprocess
#----------------------------------------------------#

#--------------- Imports for Threading --------------#
import time
import threading
#----------------------------------------------------#
import serial
import adafruit_us100
import RPi.GPIO as GPIO

#---------------- Capture and Send the First Image ----------------------#
# This is necessary to initialize the global variable "response" to the correct type.
#     "response" is used in http_msg() function, which is on a separte thread, and need 
#     to be initialized as a global variable

# Create url to send to server (using server's IP addr)
url = "http://172.20.10.12:2020/send_image"   

subprocess.run(["raspistill", "-w", "640", "-h", "480", "-t", "1", "-o", "test.jpg"])
with open("test.jpg", "rb") as f:
    image_data = f.read()

# Send the image via HTTP POST -----------------------
headers = {"Content-Type": "image/jpeg"}  # Indicate JPEG format
response = requests.post(url, data=image_data, headers=headers)
time.sleep(1) 
new_response = True

def buzzer():
        # Set the GPIO pin number
        left_buzzer_pin = 23
        right_buzzer_pin = 22
        main_buzzer_pin = 27
        buzzer_pin = left_buzzer_pin
        both_buzzer_flag = 0

        # Set the GPIO pin as output
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(left_buzzer_pin, GPIO.OUT)
        GPIO.setup(right_buzzer_pin, GPIO.OUT)

        # Set up ultrasonic sensor
        uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1) # Need to configure serial port! (Refer to links in IoT tab group)
        us100 = adafruit_us100.US100(uart)
        obj_dist = us100.distance

        while True:
                global new_response
                if new_response == True:
                        global response
                        # ----------------------Process response from server---------------------------
                        # Parse the JSON file
                        # Message1 : image upload confirmation message
                        # Message2 : command left or right, ouput by ML model
                        message = response.json()
                
                # Check the response status code, if 200 then proceed
                if response.status_code == 200:
                        if message["message2"] == "left":
                                buzzer_pin = left_buzzer_pin
                        elif message["message2"] == "right":
                                buzzer_pin = right_buzzer_pin
                        elif message ["message2"] == "middle":
                                both_buzzer_flag = 1
                        if both_buzzer_flag == 1:
                                GPIO.output(left_buzzer_pin, GPIO.HIGH)
                                GPIO.output(right_buzzer_pin, GPIO.HIGH)
                                time.sleep(0.5)
                                GPIO.output(left_buzzer_pin, GPIO.LOW)
                                GPIO.output(right_buzzer_pin, GPIO.LOW)
                        else:
                                GPIO.output(buzzer_pin, GPIO.HIGH)
                                time.sleep(0.5)
                                GPIO.output(buzzer_pin, GPIO.LOW)
                        obj_dist = us100.distance
                        print("Distance: ", obj_dist)
                        if obj_dist <= 100:
                                GPIO.output(main_buzzer_pin, GPIO.HIGH)
                                time.sleep(0.5)
                                GPIO.output(main_buzzer_pin, GPIO.LOW)

                else:
                        print("Error uploading image:", response.status_code)
                        
                new_response = False




#---------------------------------- Main --------------------------------------------#
if __name__ == '__main__':
        #-------------------- Start a New Thread for led_pot() ------------------#
        thread = threading.Thread(target=buzzer) # Spawn a thread to run buzzer() in a separate thread 
        thread.daemon = True # kill the thread as soon as the main program exit
        thread.start() # start the thread executing
        #---------------- Capture and Send the First Image ----------------------#
        while True:
            # Capture the image -----------------------------------
            # Start camera
            subprocess.run(["raspistill", "-w", "640", "-h", "480", "-t", "1", "-o", "test.jpg"])
            with open("test.jpg", "rb") as f:
                    image_data = f.read()
            
            # Send the image via HTTP POST -----------------------
            headers = {"Content-Type": "image/jpeg"}  # Indicate JPEG format
            response = requests.post(url, data=image_data, headers=headers)
            print(response)
            time.sleep(1) 
            
            # Update flag to check for new command
            new_response = True
        #-----------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

        
