# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:23:56 2024

@author: Mukhtar
"""
import pandas as pd
import os
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import font
import speech_recognition
from bidi.algorithm import get_display
from keras.models import model_from_json
import arabic_reshaper
from sklearn.feature_extraction.text import TfidfVectorizer
from text_and_voice_processing import *
from features import get_spectrogram

# Loading model Quran Surah Data
if os.path.isfile('surah.plk'):
    df = pd.read_pickle('surah.plk')
else:
    exec(open('Prepare_dataFrame.py').read())  
    df = pd.read_pickle('surah.plk')

# Loading model of Reciters Data
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('model.weights.h5')    

def voiceReco():
    recognizer = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        audio = recognizer.listen(mic)
        text = recognizer.recognize_google(audio, language='ar-AR')
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        ayah_info = run_tfidf(text,df)  
        textF.delete("1.0", "end")
        textF.insert(END, bidi_text + '\n' + ayah_info)
        textF.tag_add("center", 1.0, "end")

def browseFile():
    file_path = filedialog.askopenfilename()
    spec = get_spectrogram(file_path)
    X = []
    X.append(spec)
    X = np.array(X)
    y_pred = loaded_model.predict(X)
    predicted_class = np.argmax(y_pred, axis=1)
    path = 'Dataset'
    my_list = os.listdir(path)
    textF.delete("1.0", "end")
    s= my_list[predicted_class[0]].replace('_', ' ')
    textF.insert(END,':القاريء هو \n' + s )
    textF.tag_add("center", 1.0, "end")
    



root = Tk()
root.geometry("500x300")
root.title("التعرف على الكلام")

ButtonFont = font.Font(size=20)
LabelFont = font.Font(size=15)

Label(root, text="النص سوف يظهر هنا", font=LabelFont).pack()

textF = Text(root, height=5, width=52, font=LabelFont)
textF.tag_configure("center", justify='center')
textF.pack()

listen = Button(root, text='استمع', font=ButtonFont, command=voiceReco).place(x=170, y=200)
browse = Button(root, text='إختر الملف', font=ButtonFont, command=browseFile).place(x=270, y=200)
#listen.pack(side='left')
#browse.pack(side='right')

root.mainloop()

