# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:23:56 2024

@author: lenovo
"""
import pandas as pd
import os
from tkinter import *
from tkinter import font
import speech_recognition
from bidi.algorithm import get_display
import arabic_reshaper
from sklearn.feature_extraction.text import TfidfVectorizer
from text_and_voice_processing import *

if os.path.isfile('surah.plk'):
    df = pd.read_pickle('surah.plk')
else:
    exec(open('Prepare_dataFrame.py').read())  
    df = pd.read_pickle('surah.plk')

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

root = Tk()
root.geometry("500x300")
root.title("التعرف على الكلام")

ButtonFont = font.Font(size=20)
LabelFont = font.Font(size=15)

Label(root, text="النص سوف يظهر هنا", font=LabelFont).pack()

textF = Text(root, height=5, width=52, font=LabelFont)
textF.tag_configure("center", justify='center')
textF.pack()

Button(root, text='استمع', font=ButtonFont, command=voiceReco).place(x=220, y=200)


root.mainloop()