import serial
import adafruit_us100
import time
from gpiozero import Buzzer

uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1) # Need to configure serial port! (Refer to links in IoT tab group)
us100 = adafruit_us100.US100(uart)
buzzer = Buzzer(17) # Represents GPIO17

while True:
  obj_dist = us100.distance
  print("Distance: ", obj_dist)
  time.sleep(0.5)
  if obj_dist < 100:
    buzzer.beep()
  if obj_dist > 100:
    buzzer.off()
    
