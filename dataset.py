# for more information read "19-Intro2ML-HodaDataset.ipynb"
import cv2
import numpy as np
from scipy import io

def load_hoda(training_sample_size=1000, test_sample_size=200):
    #load dataset
    trs = training_sample_size
    tes = test_sample_size
    dataset = io.loadmat('./dataset/Data_hoda_full.mat')

    #test and training set
    X_train_orginal = np.squeeze(dataset['Data'][:trs])
    y_train = np.squeeze(dataset['labels'][:trs])
    X_test_original = np.squeeze(dataset['Data'][trs:trs+tes])
    y_test = np.squeeze(dataset['labels'][trs:trs+tes])

    #resize
    X_train_5by5 = [cv2.resize(img, dsize=(5, 5)) for img in X_train_orginal]
    X_test_5by_5 = [cv2.resize(img, dsize=(5, 5)) for img in X_test_original]
    #reshape
    X_train = [x.reshape(25) for x in X_train_5by5]
    X_test = [x.reshape(25) for x in X_test_5by_5]
    
    return X_train, y_train, X_test, y_test