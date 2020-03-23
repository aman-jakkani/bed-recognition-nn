#framer program
# 
# all imports 
import cv2    
import math  
import matplotlib.pyplot as plt    
import pandas as pd
from keras.preprocessing import image 
import numpy as np   
from keras.utils import np_utils
from skimage.transform import resize   

frame_count = 0
videoFiles = ['test1.mp4', 'test2.mp4', 'test3.mp4', 'test4.mp4', 'test5.mp4']

for each in videoFiles:
    cap = cv2.VideoCapture(each)
    frameRate = cap.get(5)
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(math.floor(frameRate)/2) == 0):
            filename ="frame%d.jpg" % frame_count;frame_count+=1
            cv2.imwrite(filename, frame)
    cap.release()
print("Done!")