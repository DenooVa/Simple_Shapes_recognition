# importings
import tkinter
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib.pyplot as plt
import numpy as np
from numba import jit,cuda
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    # width_shift_range=.15,
    # height_shift_range=.15,
    vertical_flip=True
    )
train_data_gen = image_generator.flow_from_directory(
    directory='/home/denova/Documents/py_projects/HW2/train',
    target_size=(28,28),
    batch_size=362,
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
    keras.layers.Dense(5, activation='softmax')
])
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
class_names = list(train_data_gen.class_indices.keys())
@jit(target="cuda")
def train():
    history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= train_data_gen.n // 10,
    epochs= 20
    )
train()
test = image_generator.flow_from_directory(
    directory='/home/denova/Documents/py_projects/HW2/test',
    target_size = (28,28),
    batch_size=1
)
prediction = model.predict(test)
if np.argmax(prediction[0]) == 0:
    print('circle')  
elif np.argmax(prediction[0]) == 1:
    print('inf')
elif np.argmax(prediction[0]) == 2:
    print('pentagon')
elif np.argmax(prediction[0]) == 3:
    print('square')
elif np.argmax(prediction[0]) == 4:
    print('triangle')
# # visualize
# acc = history.history['accuracy']
# loss = history.history['loss']
# epochs_range = range(20)
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training Accuracy')
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.legend(loc='upper right')
# plt.title('Training Loss')
# plt.show()