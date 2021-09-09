# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : yoloface.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import cv2
import numpy as np
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import os

from utils import *

#### 파라미터 ####
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

# model init
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#parameter  설정
SZ = 28
frame_width = 300
frame_height = 300
frame_resolution = [frame_width, frame_height]
frame_rate = 30
camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
time.sleep(0.1)

def _main():

    output_file = ''
    
    
    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):

      
        frame = frame.array
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(get_outputs_names(net))

        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)

        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        
        cv2.imshow("yolo_faceRecognition", frame)

        key = cv2.waitKey(1)
        rawCapture.truncate(0)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
