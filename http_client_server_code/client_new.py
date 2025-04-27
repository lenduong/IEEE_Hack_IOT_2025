#--------- Import for Flask/ HTTP Functions ---------#
import requests
#----------------------------------------------------#

#----------- Imports to call system command -----------#
import subprocess
#----------------------------------------------------#

#--------------- Imports for Threading --------------#
import time
# import threading
#----------------------------------------------------#

#---------------- Capture and Send the First Image ----------------------#
# This is necessary to initialize the global variable "response" to the correct type.
#     "response" is used in http_msg() function, which is on a separte thread, and need 
#     to be initialized as a global variable

# Create url to send to server (using server's IP addr)
url = "http://172.20.10.12:2020/send_image"        

#----------------------------------- Main --------------------------------------------#
if __name__ == '__main__':
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

        
