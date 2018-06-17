from imutils import paths
import numpy as np
import imutils
import cv2
from random import randint


def preprocessing(image):

    # Load the image and convert it to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT)

    # threshold the image (convert it to pure black and white)
    ret, thresh = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY)

    erodation = cv2.erode(thresh, (17, 17), iterations=1)
    dilation = cv2.bilateralFilter(erodation, 3, 75, 75)
    denoised = cv2.fastNlMeansDenoising(dilation, None, 15, 15, 7)
    dilation = cv2.dilate(denoised, (31, 31), iterations=1)
    ret, thresh = cv2.threshold(dilation, 10, 255, cv2.THRESH_BINARY)

    # Return the annotated image
    return thresh


