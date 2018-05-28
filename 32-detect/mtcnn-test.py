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

for image_path in image_paths:

    print(image_path)

    img_orig = cv2.imread(image_path)
    img = np.copy(img_orig)
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    for bounding_box in bounding_boxes:
        pts = bounding_box[:4].astype(np.int32)
        pt1 = (pts[0], pts[1])
        pt2 = (pts[2], pts[3])
        cv2.rectangle(img, pt1, pt2, color=default_color, thickness=default_thickness)

    for i in range(points.shape[1]):
        pts = points[:, i].astype(np.int32)
        for j in range(pts.size // 2):
            pt = (pts[j], pts[5 + j])
            cv2.circle(img, center=pt, radius=1, color=default_color, thickness=default_thickness)

    separator = np.zeros((img_orig.shape[0], 20, 3), np.uint8)
    cv2.imshow(image_path, np.hstack((img_orig, separator, img)))
    cv2.moveWindow(image_path, 50, 50)
    cv2.waitKey(0)
    cv2.destroyWindow(image_path)

    # cv2.imwrite('output-' + image_path, img)
