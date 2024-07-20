# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:45:05 2024

@author: lenovo
"""
import os
from pydub import AudioSegment
from pydub.utils import make_chunks


path = "C:/Users/ziada/test111/Dataset"

def process_sudio(folder_path,file,myaudio):
    #myaudio = AudioSegment.from_file(file_name, "wav") 
    # The duration of wav file
    chunk_length_ms = 20000 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    for i, chunk in enumerate(chunks): 
        chunk_name = folder_path +"/"+ file + "_{0}.wav".format(i) 
        #print ("exporting", chunk_name) 
        chunk.export(chunk_name, format="wav") 
        # if i==50: 
        #     b reak
flag = 1
for i, folder in enumerate(os.listdir(path)):
    folder_path = os.path.join(path, folder)
    print(folder)
    #print(flag)
    # if folder == 'توفيق النوري':
    #     flag = 1
    if flag == 1:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            f,e = os.path.splitext(file)
            sound = AudioSegment.from_mp3(file_path)
            process_sudio(folder_path,f,sound)
            #os.remove(os.path.join(folder_path, file)) 
            #sound.export(f'{folder_path}/0{i}.wav', format="wav")
    
    # test = os.listdir(folder_path)
    # for item in test:
    #     if item.endswith(".mp3"):
    #         os.remove(os.path.join(folder_path, item))    
        
        
# t1 = t1 * 1000 #Works in milliseconds
# t2 = t2 * 1000
# newAudio = AudioSegment.from_wav("oldSong.wav")
# newAudio = newAudio[t1:t2]
# newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.