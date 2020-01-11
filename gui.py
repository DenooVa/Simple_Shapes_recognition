from tkinter import *
import PIL
from PIL import Image, ImageDraw
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from tkinter import messagebox
import os
model = load_model('model.h5')
image_generator = ImageDataGenerator(
    rescale=1./255
    )
def save_test():
    global image_number
    filename = 'image.png'   # image_number increments by 1 at every save 
    image1.save('test/test/image.png','png')
    test = image_generator.flow_from_directory(
        directory='test',
        target_size = (28,28),
        batch_size=1,
        color_mode='rgb'
    )
    prediction = model.predict(test)
    if np.argmax(prediction[0]) == 0:
        messagebox.showinfo('Line','The shape you have drawn is a Line!')
        os.remove('test/test/image.png')  
    elif np.argmax(prediction[0]) == 1:
        messagebox.showinfo('circle','The shape you have drawn is a circle!')
        os.remove('test/test/image.png')
    elif np.argmax(prediction[0]) == 2:
        messagebox.showinfo('infinity symbol','The shape you have drawn is an infinity symbol')
        os.remove('test/test/image.png')    
    elif np.argmax(prediction[0]) == 3:
        messagebox.showinfo('star','The shape you have drawn is a pentagon star!')
        os.remove('test/test/image.png')
    elif np.argmax(prediction[0]) == 4:
        messagebox.showinfo('square','The shape you have drawn is a square!')
        os.remove('test/test/image.png')
    elif np.argmax(prediction[0]) == 5:
        messagebox.showinfo('triangle','The shape you have drawn is a triangle!')
        os.remove('test/test/image.png')
def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y
def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=10)
    draw.line((lastx, lasty, x, y), fill='black', width=10)
    lastx, lasty = x, y
root = Tk()
lastx, lasty = None, None
image_number = 0
cv = Canvas(root, width=480, height=480, bg='white')
image1 = PIL.Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(image1)
cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)
test_button = Button(root,text='Tell me what is this?',command=save_test)
def delete_items():
    cv.delete('all')
clear_button= Button(root,text='clear canvas',command=delete_items)
test_button.pack()
clear_button.pack()
root.mainloop()