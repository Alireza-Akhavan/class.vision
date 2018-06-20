###  Computer Vision and Machine Learning

# SRU:Places Recognizer Project

**Mahya Mahdian** _and_
**Mohammad Hassan Sattarian**

---

This Project aims to train a model able of recognizing six different places in our university (SRU).

Chosen places (Model Classes) :

- Computer Campus
- Architecture Campus
- Field
- buffet
- Self
- Culture house

## Quick Start

[Download the APP](https://mhsattarian.com/projects/Places%20Recognizer.apk) connect to server bellow and take picture or choose from gallery to predict:

```html
http://http://95.211.250.100:8080/predict
```

OR

Clone the repository open a terminal and enter:

```bash
cd "Usage Sample"
python predict.py [ImagesAddress]
```

## Structure

Model is a fine tuning implementation based on [VGG16: Places365](https://github.com/GKalliatakis/Keras-VGG16-places365) which is (obviously) a VGG16 network pre-trained with places images, more specifically places365 dataset, using only convolutional layers of base model with having 5 last convolutional layers unfrozen and trained connected to a 2-layered  fully-connected network with 256 and and 6 nodes, having, respectively _Relu_ and _Softmax_ as activation functions for detecting _nonlinearities_ and coding result in 6 classes.

## Dataset

Dataset used to train this model contains of 
4800 images in 6 classes, each class representing a specified place in SRU university.
images divided into approximately 3000 images for training and 1800 images for testing. each class has 500 images as train set and 300 images as test set.

for collecting dataset pictures and videos have captured from foresaid places in university from different distances, angles and times not just from the front view of the place but all around it to cover as most predict cases as possible.
images then reviewed, videos frames extracted and proper pictures selected also processed, rotated and resized to feed out model. images are resized to 108x192 pixels so the model is not that heavy and still has enough features to predict well.

## APP

For better accessibility and providing a graphical user interface (GUI) to use the model, an **Android App** is designed and created. using the App user can predict an image, taken in the app or chosen from the gallery, by writing the prediction server address. image then sent to the server and prediction result would be shown after few moments.
