# importings
import tkinter
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    vertical_flip=True,
    horizontal_flip=True
    )
test = image_generator.flow_from_directory(
    directory='test',
    target_size = (28,28),
    batch_size=1
)
prediction = model.predict(test)
if np.argmax(prediction[0]) == 0:
    print('line')  
elif np.argmax(prediction[0]) == 1:
    print('circle')
elif np.argmax(prediction[0]) == 2:
    print('inf')
elif np.argmax(prediction[0]) == 3:
    print('pentagon')
elif np.argmax(prediction[0]) == 4:
    print('square')
elif np.argmax(prediction[0]) == 5:
    print('triangle')