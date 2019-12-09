# importings
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib.pyplot as plt
import numpy as np
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(
    directory='/home/denova/Documents/py_projects/HW2/train',
    target_size=(28,28),
    batch_size=10,
    color_mode='rgb'
)
model = keras.Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(28,28,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'), # MLP part of the CNN
    keras.layers.Dense(3, activation='softmax')
])
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= train_data_gen.n // 10,
    epochs= 10
)
test = image_generator.flow_from_directory(
    directory='/home/denova/Documents/py_projects/HW2/test',
    target_size = (28,28),
    batch_size=1
)
prediction = model.predict(test)
if np.argmax(prediction[0]) == 0:
    print('circle')  
elif np.argmax(prediction[0]) == 1:
    print('square')
else:
    print('triangle')