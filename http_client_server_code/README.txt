Group: Grishma Shukla and Le Duong

CODE EXECUTION INSTRUCTION:
1. The code is run entirely in Python3.10 (or any version that works with Tensorflow)
2. Need 2 terminals, one SSH to the RPi with a camera attached to it, the other will run the server
3. The computer that runs the server needs Tensorflow to be properly installed 
4. The files structure needs to stay exactly the way this folder was uploaded (to ensure correct paths)
6. Commands (need to cd to http folder / start server first):
        Server: 
                # If this is a VM of some sort, make sure to forward port 8080 from the host machine
                #   and make sure firewall isn't blocking the port forwarding
                # Replace image path in line 79 with "<your current working directory>/uploads/test_image.jpg"
                $ python3 server.py
        Client: 
                # Replace the ip address in the URL with your server's address (lines 27 & 150)
                $ python3 client.py
7. Navigate to a web browser and type "http://localhost:8080/" to display front end web :)

LIST OF EXTERNAL LIBRARIES:
1. Flask
2. Requests
3. cv2 (OpenCV)
4. Tensorflow
5. Tensorflow Keras
6. adafruit-mcp3008
7. matplotlib
8. numpy
9. PIL (Pillow)
