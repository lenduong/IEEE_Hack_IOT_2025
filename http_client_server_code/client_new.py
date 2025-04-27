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

#---------------- Capture and Send the First Image ----------------------#
# This is necessary to initialize the global variable "response" to the correct type.
#     "response" is used in http_msg() function, which is on a separte thread, and need 
#     to be initialized as a global variable

# Create url to send to server (using server's IP addr)
url = "http://172.20.10.12:8080/send_image"
new_response = True
response = None

#---------------------- Function to check for HTTP messages ------------------------#
def http_msg():
        while True:
            global new_response
            if new_response ==True:
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
                    print("ooooooooooooooooooooooooooooooooooooooooooo")
                else:
                    print("Error uploading image:", response.status_code)
                new_response = False
#-------------------------------------------------------------------------------------#
            

#----------------------------------- Main --------------------------------------------#
if __name__ == '__main__':
        #-------------------- Start a New Thread for led_pot() ------------------#
        thread = threading.Thread(target=http_msg) # Spawn a thread to run http_msg() in a separate thread 
        thread.daemon = True # kill the thread as soon as the main program exit
        thread.start() # start the thread executing
        #------------------------------------------------------------------------#

        #---------------- Capture and Send the First Image ----------------------#
        while True:
            
            # Capture the image -----------------------------------
            # Start camera
            subprocess.run(["raspistill", "-o", "test.jpg"])
            with open("test.jpg", "rb") as f:
                    image_data = f.read()
            
            # Send the image via HTTP POST -----------------------
            headers = {"Content-Type": "image/jpeg"}  # Indicate JPEG format
            response = requests.post(url, data=image_data, headers=headers)
            # http_msg(response)
            time.sleep(3) 
            
            # Update flag to check for new command
            new_response = True
        #-----------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

        
