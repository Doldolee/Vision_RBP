from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import tensorflow as tf
import time

model = tf.keras.models.load_model('digits_model.h5')
SZ = 28
frame_width = 300
frame_height = 300
frame_resolution = [frame_width, frame_height]
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    cv2MajorVersion = cv2.__version__.split(".")[0]
    if int(cv2MajorVersion) >= 4:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_digits = []
    positions = []

    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)

      # Ignore small sections
      if w * h < 2400: continue
      y_position = y-margin
      if(y_position < 0): y_position = 0
      x_position = x-margin
      if(x_position < 0): x_position = 0
      img_roi = thresh[y_position:y+h+margin, x_position:x+w+margin]
      num = cv2.resize(img_roi, (SZ,SZ))
      num = num.astype('float32') / 255.
      
      result = model.predict(np.array([num]))
      result_number = np.argmax(result)
      cv2.rectangle(image, (x-margin, y-margin), (x+w+margin, y+h+margin), (0, 255, 0), 2)
      
      text = "Number is : {} ".format(result_number)
      cv2.putText(image, text, (margin, frame_height-margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    # show the frame
    cv2.imshow("MNIST Hand Write", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break