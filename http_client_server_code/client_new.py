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
import RPi.GPIO as GPIO
import time

#---------------- Capture and Send the First Image ----------------------#
# This is necessary to initialize the global variable "response" to the correct type.
#     "response" is used in http_msg() function, which is on a separte thread, and need 
#     to be initialized as a global variable

def buzzer():
        # Set the GPIO pin number
        left_buzzer_pin = 23
        right_buzzer_pin = 22

        # Set the GPIO pin as output
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(left_buzzer_pin, GPIO.OUT)
        GPIO.setup(right_buzzer_pin, GPIO.OUT)

        while True:
                # Turn on BUZZER based on Midas ()
                # Then 
                GPIO.output(buzzer_pin, GPIO.HIGH) # Turn buzzer on
                time.sleep(1)
                GPIO.output(buzzer_pin, GPIO.LOW) # Turn buzzer off
                time.sleep(1)

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


# Create url to send to server (using server's IP addr)
url = "http://172.20.10.12:2020/send_image"   


#---------------------------------- Main --------------------------------------------#
if __name__ == '__main__':
        #-------------------- Start a New Thread for led_pot() ------------------#
        thread = threading.Thread(target=buzzer) # Spawn a thread to run led_pot() in a separate thread 
        thread.daemon = True # kill the thread as soon as the main program exit
        thread.start() # start the thread executing
        #------------------------------------------------------------------------#
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

        
