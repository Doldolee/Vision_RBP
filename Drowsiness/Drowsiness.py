from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import tensorflow as tf
import time
import pygame

pygame.mixer.init()
pygame.mixer.music.load('fire-truck.wav')

face_cascade_name = './haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = './haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

model = tf.keras.models.load_model('drowsinessDetection.h5')

SZ = 24
status = 'Awake'
number_closed = 0
closed_limit = 7
show_frame = None
sign = None
color = None

frame_width = 640
frame_height = 480
frame_resolution = [frame_width, frame_height]
frame_rate = 16

camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start_time = time.time()
    image = frame.array
    show_frame = image
    height,width = image.shape[:2]
    #튜닝
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        show_frame = cv2.rectangle(show_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #얼굴 영역만 자름
        faceROI = frame_gray[y:y+h,x:x+w]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        results = []

        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            show_frame = cv2.circle(show_frame, eye_center, radius, (0, 255, 255 ), 2)
            eye = faceROI[y2:y2+h2,x2:x2+w2]
            eye = cv2.resize(eye,(SZ,SZ))
            eye= eye/255
            eye=  eye.reshape(SZ,SZ,-1)
            eye = np.expand_dims(eye,axis=0)
            result = model.predict_classes(eye)
            results.append(result[0])
        print(result[0], results, np.mean(results))
        #두 눈을 다 떳을 경우에만 awake
        if( np.mean(results)==1 ):
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 1
            if( number_closed<0 ):
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'Sleep'
            number_closed = number_closed + 1
            
        sign = status + ', Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)
        if( number_closed > closed_limit ):
            show_frame = frame_gray
            
        
    cv2.putText(show_frame, sign , (10,height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Drowsiness Detection", show_frame)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
