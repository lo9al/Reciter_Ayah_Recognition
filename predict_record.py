# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:17:02 2024

@author: lenovo
"""

import numpy as np
import os
from keras.models import model_from_json
from features import get_spectrogram
import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(duration, fs):
    print(f'Recording for {duration} seconds...')
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording

def save_recording(recording, filename, fs):
    wav.write(filename, fs, recording)
    
duration = 2
sample_rate = 22050
recording_filename='recording.wav'   

print('Start Recording')
recorded_audio = record_audio(duration, sample_rate)
save_recording(recorded_audio,recording_filename,sample_rate)
print('Record End...')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.weights.h5")


spec = get_spectrogram(recording_filename)
X = []
X.append(spec)
X = np.array(X)
y_pred = loaded_model.predict(X)
predicted_class = np.argmax(y_pred, axis=1)
path = 'Dataset'
my_list = os.listdir(path)
print(my_list[predicted_class[0]])