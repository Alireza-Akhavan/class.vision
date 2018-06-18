import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
import matplotlib.pyplot as plt
from digit import load_img

np.random.seed(123)  # for reproducibility


x_train_original, y_train_original, x_test_original, y_test_original = load_img()


# Preprocess input data for Keras.
x_train = np.array(x_train_original)
y_train = keras.utils.to_categorical(y_train_original, num_classes=10)
x_test = np.array(x_test_original)
y_test = keras.utils.to_categorical(y_test_original, num_classes=10)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(-1,20,20,1)
x_test = x_test.reshape(-1,20,20,1)

#-----------------------------------------------------------------

# Settings :
EPOCH_NO = 64
BATCH_SIZE = 64
OPTIMIZER = 'rmsprop'
METRICS = 'accuracy'


train_acc = []
train_loss = []
test_acc = []
test_loss = []

# Create model :
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(20, 20, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (1, 1), activation='relu'))
model.add(layers.Flatten())
model.add(Dropout(0.8))
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.8))
model.add(layers.Dense(10, activation='softmax'))


# Compile model :
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=[METRICS])



# Fit model :

for i in range(EPOCH_NO):
	model.fit(x_train, y_train,epochs=1,batch_size=BATCH_SIZE,validation_data = (x_test, y_test))
	loss_te,acc_te = model.evaluate(x_test, y_test)
	loss_tr,acc_tr = model.evaluate(x_train, y_train)
	train_loss.append(loss_tr)
	train_acc.append(acc_tr)
	test_loss.append(loss_te)
	test_acc.append(acc_te)


plt.figure(1)
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Test')
plt.ylabel('Cost Function')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(train_acc, label='Train')
plt.plot(test_acc, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Save Model :
model.save('captcha.h5')  # creates a HDF5 file 'my_model.h5'