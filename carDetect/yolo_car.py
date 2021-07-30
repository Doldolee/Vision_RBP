import cv2
import numpy as np
import time

min_confidence = 0.5
margin = 30
file_name = "image/parking_02.jpg"

net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

classes = []

with open("coco.names","r")as f:
  classes = [line.strip() for line in f.readlines()]
print(classes)

layer_names = net.getLayerNames()
ouput_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

start_time = time.time()
img = cv2.imread(file_name)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(ouput_layers)


confidences = []
boxes = []

for out in outs:
  for detection in out:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    if class_id ==2 and confidence > min_confidence:
      center_x = int
      center_x = int(detection[0] * width)
      center_y = int(detection[1] * height)
      w = int(detection[2] * width)
      h = int(detection[3] * height)

      
      x = int(center_x - w / 2)
      y = int(center_y - h / 2)

      boxes.append([x, y, w, h])
      confidences.append(float(confidence))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
color = (0, 255, 0)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = '{:,.2%}'.format(confidences[i])
        print(i, label)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

text = "Number of Car is : {} ".format(len(indexes))
cv2.putText(img, text, (margin, margin), font, 2, color, 2)

cv2.imshow("Number of Car - "+file_name, img)

end_time = time.time()
process_time = end_time - start_time
print("=== A frame took {:.3f} seconds".format(process_time))


import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("rbpcar-18172-firebase-adminsdk-31mev-fce0257542.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rbpcar-18172-default-rtdb.firebaseio.com/',
    "storageBucket":"rbpcar-18172.appspot.com"
})

bucket = storage.bucket()

blob = bucket.blob(file_name)

blob.upload_from_filename(file_name)

ref = db.reference('parking')
box_ref = ref.child('west-coast')
box_ref.update({
    'count': len(indexes),
    'time': time.time(),
    'image': blob.public_url
})

cv2.waitKey(0)
cv2.destroyAllWindows()
