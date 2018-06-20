import argparse

import numpy as np
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image
from keras.models import load_model

from places_utils import \
    preprocess_input  # places_utiles file is provided by VGG16:places contributors.

parser = argparse.ArgumentParser()
parser.add_argument("ImageAddress", help="Address of the images to predict.")
args = parser.parse_args()
ImageAddress = args.ImageAddress

model = load_model('../Model/SRU_Places_6.h5')

# labels ordered corresponding to recognized classes by model.
# extracted from --> label_map = (train_generator.class_indices)
labels = ['Architect Campus', 'Buffet', 'Computer Campus', 'Culture house', 'Field', 'Self']


# predicting
img = image.load_img(ImageAddress, target_size=(108, 192))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# printing prediction 
prediction = model.predict(x)
# y_classes = y_prob.argmax()     # -->  uncomment for printing it's most probable Class Number
# y_true_labels = train_generator.classes    # --> uncomment for printing each train image Class Number
print (prediction)

# Printing each class probability
for i, p in enumerate(prediction[0]):
  print('%s Probability: \t %f' % (labels[i], p))
