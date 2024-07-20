# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:39:37 2024

@author: lenovo
"""
import numpy as np 
import os
import librosa
import librosa.display
from keras.utils import to_categorical
from skimage.transform import resize

sr = 22050
duration = 5

# Set the size of the spectrogram images
img_height = 128
img_width = 256

# Define a function to extract the spectrogram from an audio file
def get_spectrogram(file_path):
    # Load the audio file
    signal, sr = librosa.load(file_path, sr=22050, duration=duration)
    # Compute the spectrogram
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    # Convert the spectrogram to dB scale
    spec_db = librosa.power_to_db(spec, ref=np.max)
    # Resize the spectrogram to the desired shape
    spec_resized = librosa.util.fix_length(spec_db, size = duration * sr // 512 + 1)
    spec_resized = resize(spec_resized, (img_height, img_width), anti_aliasing=True)
    return spec_resized


# Define a function to load the dataset
def load_dataset(path):
    X = []
    y = []
    for i, folder in enumerate(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # Extract the spectrogram from the audio file
            spec = get_spectrogram(file_path)
            X.append(spec)
            y.append(i)
    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # Convert the labels to one-hot encoding
    y = to_categorical(y)
    return X, y