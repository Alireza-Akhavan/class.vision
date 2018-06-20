'''************   Computer Vision   *************'''
'''         Project : Places Recognition         '''
'''                     ***                      '''
'''            Mahya Mahdian - 94471039          '''
'''      Mohammad Hassan Sattarian - 94471035    '''


import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, optimizers
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from places_utils import preprocess_input
from vgg16_places_365 import VGG16_Places365

# Model Used as Base model in Fine Tuning
base_model = VGG16_Places365(include_top=False, weights='places', input_shape=(108, 192, 3))

# making 5 last layers *Unfreeze*
for layer in base_model.layers[:12]:
  layer.trainable = True
for layer in base_model.layers[12:]:
  layer.trainable = False

# Creating out very own model based on VGG616:Places365 model with additional
# fully-connected layers containing 256 and 6 nodes, having, respectively Relu and Softmax as activation
# functions for detecting nonlinearities and coding result in 6 classes
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))


# base model and model summeries are printed. mainly to check trainable data numbers
base_model.summary()
model.summary()


# data agumentation methods are used to achive better result; methods to augument images selected 
# according to real-life situations to simulate real cases.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

# data generators are used to feed the network from **Train** and **Test** directories.
train_dir = 'Train'
validation_dir = 'Test'

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 108x192
        target_size=(108, 192),
        batch_size=20,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(108, 192),
        batch_size=20,
        class_mode='categorical')



# Compiling the model with *categorical_crossentropy* loss 'cause our problem is categorical 
# using RMSProp optimizer and *accuracy* metrics; then fitting it with created generators
# within 30 epochs. (this step would take some time!)
model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=2e-5),
            metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=149,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=90)


''' For showing visualizations of model
    including *Training and validation accuracy* and 
    *Training and validation loss* plots
    uncomment following lines '''

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()


# Saving model for further use.
model.save('SRU_Places_6.h5')
