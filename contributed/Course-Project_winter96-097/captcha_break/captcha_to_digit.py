from imutils import paths
import numpy as np
import imutils
import cv2
from random import randint
from contour import number_to_digit
from dataset import load_img

# Counter :
k = 0

# Open label file :
f = open('./digits/labels.txt', 'w')


# Load preprocessed image :
images, labels = load_img(training_sample_size=3000, test_sample_size=0, x_size=90, y_size=25, all=True)
#new_image = cv2.imread('./dataset/%d.png' % randint(0, 1000))   #Random import
print(images.shape,labels)

for image,(i,label) in zip(images,enumerate(labels)):

    # Seprate numbers :
    seprated_numbers = number_to_digit(image)

    # Create digit image :
    if seprated_numbers != False:
        for digit , (j,label_digit) in zip(seprated_numbers,enumerate(list(label))):
            # Save image :
            cv2.imwrite( "./digits/"+str(k+j)+".png",digit)
            # Save label :
            f.write('%s\n' %label_digit)
    else:
        k -= 5

    # Counter :
    k += 5

# Close file :
f.close()  # you can omit in most cases as the destructor will call it