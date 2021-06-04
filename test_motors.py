import RPi.GPIO as GPIO
from time import sleep
 
GPIO.setmode(GPIO.BCM)  
Motor1A = 16
Motor1B = 26
 
Motor2A = 20
Motor2B = 21
 
GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
 
GPIO.setup(Motor2A,GPIO.OUT)
GPIO.setup(Motor2B,GPIO.OUT)
 
print("Going forwards")
GPIO.output(Motor1A,GPIO.HIGH)
GPIO.output(Motor1B,GPIO.LOW)
 
GPIO.output(Motor2A,GPIO.HIGH)
GPIO.output(Motor2B,GPIO.LOW)

GPIO.cleanup()
