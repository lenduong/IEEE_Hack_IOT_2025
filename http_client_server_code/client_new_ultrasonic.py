import serial
import time
import adafruit_us100
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1)
us100 = adafruit_us100.US100(uart)

while True:
  obj_dist = us100.distance
  print("Distance: ", obj_dist)
  time.sleep(0.5)
