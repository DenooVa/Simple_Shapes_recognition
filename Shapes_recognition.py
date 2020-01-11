# importings
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = ImageDataGenerator(
    rescale=1./255,
    # data augmentation
    rotation_range=45,
    vertical_flip=True,
    horizontal_flip=True
    )
train_data_gen = image_generator.flow_from_directory(
    directory='train',
    target_size=(28,28),
    batch_size=204,
    color_mode='rgb'
    )
model = keras.Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(28,28,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'), # MLP part of the CNN
    keras.layers.Dense(6, activation='softmax')
])
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit_generator(
        train_data_gen,
        steps_per_epoch= train_data_gen.n // 10,
        epochs= 30
        )
#saving
model.save('model.h5')
model.save_weights('model_weights.h5')
# visualize
acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(20)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()