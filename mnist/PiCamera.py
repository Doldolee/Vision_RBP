from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import time

frame_width = 300
frame_height = 300
frame_resolution = [frame_width,frame_height]
frame_rate = 16
margin = 30

camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
time.sleep(0.1)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  image = frame.array
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hue, saturation, value = cv2.split(hsv)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
  blackhat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
  add = cv2.add(value, topHat)
  subtract = cv2.subtract(add, blackhat)
  blur = cv2.GaussianBlur(subtract, (5,5),0)
  thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
  cv2.imshow('thresh', thresh)

  key = cv2.waitKey(1)&0xFF
  rawCapture.truncate(0)

  if key == ord("q"):
    break