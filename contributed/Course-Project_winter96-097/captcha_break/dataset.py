# Load captcha img :
import cv2
import numpy as np
from preprocess import preprocessing


def load_img(training_sample_size=1000, test_sample_size=200, x_size=18, y_size=5, all=False):
    #load dataset
	dataset = dict()
	dataset['labels'] = [line.rstrip('\n') for line in open('./dataset/labels.txt')]
	
	new_image = cv2.imread('./dataset/%d.png' %0)
	# Preprocessing :
	new_image = preprocessing(new_image)
	images = np.array([new_image])


	
	
	for i in range(1,training_sample_size + test_sample_size):
		new_image = cv2.imread('./dataset/%d.png' %i)
		#Preprocessing :
		new_image = preprocessing(new_image)
		images = np.insert(images, i, [new_image], 0)

	if (all):
		return images,dataset['labels'][:training_sample_size + test_sample_size]

    #test and training set
	X_train_orginal = images[:training_sample_size]
	y_train = np.squeeze(dataset['labels'][:training_sample_size])
	X_test_original = images[training_sample_size:training_sample_size+test_sample_size]
	y_test = np.squeeze(dataset['labels'][training_sample_size:training_sample_size+test_sample_size])
	
    
	#resize
	X_train_5by5 = [cv2.resize(img, dsize=(x_size, y_size)) for img in X_train_orginal]
	X_test_5by_5 = [cv2.resize(img, dsize=(x_size, y_size)) for img in X_test_original]
    #reshape
	X_train = [x.reshape(x_size*y_size) for x in X_train_5by5]
	X_test = [x.reshape(x_size*y_size) for x in X_test_5by_5]

    #return
	return X_train, y_train, X_test, y_test
	
