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

        # Set the GPIO pin as output
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(left_buzzer_pin, GPIO.OUT)
        GPIO.setup(right_buzzer_pin, GPIO.OUT)
        GPIO.setup(main_buzzer_pin, GPIO.OUT)
        left_pwm = GPIO.PWM(left_buzzer_pin, 500)  # 1kHz
        right_pwm = GPIO.PWM(right_buzzer_pin, 1000)  # 1kHz
        main_pwm = GPIO.PWM(main_buzzer_pin, 1500)  # 1kHz

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
                        print(message)
                        if message["message2"] == "left":
                                left_pwm.start(50)  # 50% duty cycle for sound
                                time.sleep(0.5)
                                left_pwm.stop()
                        elif message["message2"] == "right":
                                right_pwm.start(50)  # 50% duty cycle for sound
                                time.sleep(0.5)
                                right_pwm.stop()
                        elif message ["message2"] == "middle":
                                left_pwm.start(50)  # 50% duty cycle for sound
                                time.sleep(0.5)
                                left_pwm.stop()
                                right_pwm.start(50)  # 50% duty cycle for sound
                                time.sleep(0.5)
                                right_pwm.stop()
                        obj_dist = us100.distance
                        print("Distance: ", obj_dist)
                        if obj_dist <= 100:
                                main_pwm.start(50)
                                time.sleep(0.5)
                                main_pwm.stop()

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

        
