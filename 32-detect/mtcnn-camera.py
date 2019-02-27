# *-* coding: utf-8 *-*

import tensorflow as tf
import numpy as np
import cv2
import os
import re
import detect_face

default_color = (0, 255, 0) #BGR
default_thickness = 2

image_paths = sorted([f for f in os.listdir('.') if re.match(r'.+\.jpg', f)])

with tf.Graph().as_default():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, img_orig = cap.read()
    img = np.copy(img_orig)

    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    for bounding_box in bounding_boxes:
        pts = bounding_box[:4].astype(np.int32)
        pt1 = (pts[0], pts[1])
        pt2 = (pts[2], pts[3])
        cv2.rectangle(img, pt1, pt2, color=default_color, thickness=default_thickness)
    cv2.imshow("salam",  img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



