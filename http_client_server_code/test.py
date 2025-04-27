import RPi.GPIO as GPIO
import time

pin = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)
pwm = GPIO.PWM(pin, 1000)  # 1kHz

while True:
  GPIO.output(pin, GPIO.HIGH)
  time.sleep(0.5)
  GPIO.output(pin, GPIO.LOW)
