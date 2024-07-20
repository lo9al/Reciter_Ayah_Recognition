# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:10:43 2024

@author: lenovo
"""

import os
import shutil

srcpath = "C:/Users/ziada/test111/Dataset"
destpath = srcpath

for root, subFolders, files in os.walk(srcpath):
    for file in files:
        splitted = file.split()
        if splitted[0]=='عبد':
            folder_name = splitted[0]+' '+splitted[1]+' '+splitted[2]
        else:     
            folder_name = splitted[0]+' '+splitted[1]
        subFolder = os.path.join(destpath,folder_name)
        if not os.path.isdir(subFolder):
             os.makedirs(subFolder)
        shutil.move(os.path.join(root, file), subFolder)
        