import serial
import adafruit_us100
import time
from gpiozero import Buzzer
from gpiozero import PWMOutputDevice

uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1) # Need to configure serial port! (Refer to links in IoT tab group)
us100 = adafruit_us100.US100(uart)
# buzzer = Buzzer(17) # Represents GPIO17
buzzer = PWMOutputDevice(17) # If buzzer is passive
buzzer.frequency = 1000  # Sets a 1000 Hz tone

while True:
  obj_dist = us100.distance
  print("Distance: ", obj_dist)
  time.sleep(0.5)
  if obj_dist < 100:
    # buzzer.on()
    buzzer.value = 0.5 # Turn on (duty cycle 50%) for passive buzzer
  if obj_dist > 100:
    buzzer.off()
    
