import serial
import adafruit_us100
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1)
us100 = adafruit_us100.US100(uart)
print("Distance: ", us100.distance)
