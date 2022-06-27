import tkinter as tk
from tkinter import filedialog
from model import *
from PIL import Image
import numpy as np
import pylab as plt

model = DigitModel()
def ready_data():
    model.ready_data()
def train():
    model.train_model(15)
def test():
    model.test_model()

def predict():
    im_arr = pre_predict()
    pred = model.predict(im_arr)
    print(pred)
    pred_label['text'] = 'Digit Is: ' + str(pred)
    pred_label.pack(fill='both', expand=True)

def pre_predict():
    filename = filedialog.askopenfilename()
    if not filename:
        return
    image = Image.open(filename)
    image = image.resize((32,32))
    image_array = np.array(image)
    im = image_array.astype(int)
    plt.imshow(im, interpolation='nearest')
    plt.show()
    
    image_array = image_array.astype(float)
    image_array = np.reshape(image_array, (1, 32, 32, 3))
    return image_array

def switch():
    model.switch_to_pretrained()

app = tk.Tk()
app.geometry('200x200')
tk.Button(app, text='Ready Data', command=ready_data).pack(fill='both', expand=True)
tk.Button(app, text='Train Model',command=train).pack(fill='both', expand=True)
tk.Button(app, text='Show Graphs', command=model.show_graphs).pack(fill='both', expand=True)
tk.Button(app, text='Test Model',command=test).pack(fill='both', expand=True)
tk.Button(app, text='Make Predictions', command=predict).pack(fill='both', expand=True)
tk.Button(app, text='Switch To Pretrained', command=switch).pack(fill='both', expand=True)
pred_label = tk.Label(app, text='')

app.mainloop()