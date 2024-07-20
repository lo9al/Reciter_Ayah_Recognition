# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:47:58 2024

@author: Mukhtar
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:17:02 2024

@author: Mukhtar
"""

import numpy as np
import os
from keras.models import model_from_json
from features import get_spectrogram
import tkinter as tk
from tkinter import filedialog

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("model.weights.h5")

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
spec = get_spectrogram(file_path)
X = []
X.append(spec)
X = np.array(X)
y_pred = loaded_model.predict(X)
predicted_class = np.argmax(y_pred, axis=1)
path = 'Dataset'
my_list = os.listdir(path)
print(my_list[predicted_class[0]])