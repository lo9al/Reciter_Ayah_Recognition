# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:23:56 2024

@author: lenovo
"""
import pandas as pd
import os
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import font
from speech_recognition.exceptions import UnknownValueError
import speech_recognition
from bidi.algorithm import get_display
from keras.models import model_from_json
import arabic_reshaper
from sklearn.feature_extraction.text import TfidfVectorizer
from text_and_voice_processing import *
from features import get_spectrogram
from pydub.playback import play
from scipy.io import wavfile
from pydub.playback import play
from scipy.io import wavfile


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
        raw = audio.get_raw_data()
        
        sample_rate = 22050  # or your specific sample rate
        audio_array = np.frombuffer(raw, dtype=np.int16)
        wavfile.write('output.wav', sample_rate, audio_array)
        spec = get_spectrogram('output.wav')
        X = []
        X.append(spec)
        X = np.array(X)
        y_pred = loaded_model.predict(X)
        #print(y_pred)
        predicted_class = np.argmax(y_pred, axis=1)
        if (y_pred[0,predicted_class[0]] > 0.6):
            path = 'Dataset'
            my_list = os.listdir(path)
            s= my_list[predicted_class[0]].replace('_', ' ')
            print(y_pred[0,predicted_class[0]])
            print(y_pred)
        else:
            s= 'غير معروف'
        #play(audio_array)
        try:
            text = recognizer.recognize_google(audio, language='ar-AR')
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            ayah_info = run_tfidf(text,df)  
        except UnknownValueError:
            ayah_info = 'لم يتم التعرف على الكلام' 
        textF.delete("1.0", "end")
        textF.insert(END,':القاريء هو \n' + s + '\n' + ayah_info)
        #textF.insert(END, bidi_text + '\n' + ayah_info)
        textF.tag_add("center", 1.0, "end")

def browseFile():
    file_path = filedialog.askopenfilename()
    # samplerate, data = wavfile.read(file_path)
    # play(data)
    r = speech_recognition.Recognizer()
    with speech_recognition.WavFile(file_path) as source:              # use "test.wav" as the audio source
        audio = r.record(source)                        # extract audio data from the file
        
        
        try:
            text = r.recognize_google(audio, language='ar-AR')
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            ayah_info = run_tfidf(text,df)  
        except UnknownValueError:
            ayah_info = 'لم يتم التعرف على الكلام' 
            
            
        # text = r.recognize_google(audio, language='ar-AR')
        # reshaped_text = arabic_reshaper.reshape(text)
        # bidi_text = get_display(reshaped_text)
        # print(bidi_text)
        # ayah_info = run_tfidf(text,df)
        # print(text)
        
    #try:
        #print("Transcription: " + r.recognize(audio))   # recognize speech using Google Speech Recognition
    #    print('')
    #except LookupError:                                 # speech is unintelligible
    #    print("Could not understand audio")
    spec = get_spectrogram(file_path)
    X = []
    X.append(spec)
    X = np.array(X)
    y_pred = loaded_model.predict(X)
    #print(y_pred)
    predicted_class = np.argmax(y_pred, axis=1)
    if (y_pred[0,predicted_class[0]] > 0.6):
        path = 'Dataset'
        my_list = os.listdir(path)
        s= my_list[predicted_class[0]].replace('_', ' ')
    else:
        s= 'غير معروف'
    #textF.insert(END,':القاريء هو \n' + s )
    textF.delete("1.0", "end")
    textF.insert(END,':القاريء هو \n' + s + '\n' + ayah_info)
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

