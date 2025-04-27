import RPi.GPIO as GPIO
import time

pin = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)
pwm = GPIO.PWM(pin, 1000)  # 1kHz

while True:
    pwm.start(50)  # 50% duty cycle for sound
    time.sleep(0.5)
    pwm.stop()
