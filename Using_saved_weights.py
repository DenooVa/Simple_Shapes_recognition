# importings
import tkinter
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
model = load_model('/home/denova/Documents/py_projects/HW2/model.h5')
image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    # width_shift_range=.15,
    # height_shift_range=.15,
    vertical_flip=True
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
    print('inf')
elif np.argmax(prediction[0]) == 2:
    print('line')
elif np.argmax(prediction[0]) == 3:
    print('pentagon')
elif np.argmax(prediction[0]) == 4:
    print('square')
elif np.argmax(prediction[0]) == 5:
    print('triangle')