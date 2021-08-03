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


# import argparse
import cv2
import numpy as np
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import os

from utils import *

#####################################################################
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

#####################################################################
# print the arguments
# print('----- info -----')
# print('[i] The config file: ', args.model_cfg)
# print('[i] The weights of model file: ', args.model_weights)
# print('[i] Path to image file: ', args.image)
# print('[i] Path to video file: ', args.video)
# print('###########################################################\n')

# check outputs directory
# if not os.path.exists(args.output_dir):
#     print('==> Creating the {} directory...'.format(args.output_dir))
#     os.makedirs(args.output_dir)
# else:
#     print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

SZ = 28
frame_width = 300
frame_height = 300
frame_resolution = [frame_width, frame_height]
frame_rate = 30
# margin = 30
camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
time.sleep(0.1)

def _main():
    # wind_name = 'face detection using YOLOv3'
    # cv2.namedWindow(wind_name, cv2.WINDOW_FREERATIO)

    output_file = ''
    # cap = cv2.VideoCapture(args.src)

    # Get the video writer initialized to save the output video
    # if not args.image:
    #     video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
    #                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
    #                                    cap.get(cv2.CAP_PROP_FPS), (
    #                                        round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):

        # has_frame, frame = cap.read()

        # Stop the program if reached end of video
        # if not has_frame:
        #     print('[i] ==> Done processing!!!')
        #     print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
        #     cv2.waitKey(1000)
        #     break

        # Create a 4D blob from a frame.

        frame = frame.array
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
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

    # cap.release()
    # cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
