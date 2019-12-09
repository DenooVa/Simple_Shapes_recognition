# importings
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# The 1./255 is to convert from uint8 to float32 in range [0,1].
# load images
# here we can manipulate this to augment accuraccy
Image_Generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,horizontal_flip=True,
                                                               rotation_range=45 , zoom_range=0.5,
                                                               width_shift_range=.15,
                                                               height_shift_range=.15)
# some default vals
BATCH_SIZE = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
epochs = 15
# generating the data we need
train_data_gen = Image_Generator.flow_from_directory(
    directory='/home/denova/Documents/py_projects/HW2/Shapes/train',
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')
# load validation data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
# generating validation data we need
val_data_gen = validation_image_generator.flow_from_directory(
    directory='/home/denova/Documents/py_projects/HW2/Shapes/validation',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

# model generator
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# summary
model.summary()

# training
history = model.fit_generator(
    train_data_gen,
    epochs=epochs,
    validation_data=val_data_gen,
)

# visualizing
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
