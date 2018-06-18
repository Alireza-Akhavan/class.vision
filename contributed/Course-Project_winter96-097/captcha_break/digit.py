# Load number img :
import cv2
import numpy as np


def load_img(training_sample_size=7000, test_sample_size=800, x_size=20, y_size=20, all=False):
    # load dataset
    dataset = dict()
    dataset['labels'] = [line.rstrip('\n') for line in open('./digits/labels.txt')]

    new_image = cv2.imread('./digits/%d.png' % 0)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    images = np.array([new_image])

    for i in range(1, training_sample_size + test_sample_size):
        new_image = cv2.imread('./digits/%d.png' % i)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        images = np.insert(images, i, [new_image], 0)

    if (all):
        return images, dataset['labels'][:training_sample_size + test_sample_size]

    # test and training set
    X_train_orginal = images[:training_sample_size]
    y_train = np.squeeze(dataset['labels'][:training_sample_size])
    X_test_original = images[training_sample_size:training_sample_size + test_sample_size]
    y_test = np.squeeze(dataset['labels'][training_sample_size:training_sample_size + test_sample_size])

    # resize
    X_train_5by5 = [cv2.resize(img, dsize=(x_size, y_size)) for img in X_train_orginal]
    X_test_5by_5 = [cv2.resize(img, dsize=(x_size, y_size)) for img in X_test_original]
    # reshape
    #X_train = [x.reshape(x_size * y_size) for x in X_train_5by5]
    #X_test = [x.reshape(x_size * y_size) for x in X_test_5by_5]

    # return
    return X_train_5by5, y_train, X_test_5by_5, y_test

