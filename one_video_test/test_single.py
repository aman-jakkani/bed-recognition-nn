#test one video at a time

# all imports 
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import json
import cv2
import os
import glob

d = os.path.dirname(os.getcwd())
model = load_model(d+'/bedmodel.h5')
#get test data frames
frame_count = 0
videoFile = 'testvideo.mp4'
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5)
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(math.floor(frameRate)/2) == 0):
        filename ="frames/frame%d.jpg" % frame_count;frame_count+=1
        cv2.imwrite(filename, frame)
cap.release()

test_imgs = []
for i in range(0, frame_count):
    img = plt.imread('frames/frame'+str(i)+'.jpg')
    test_imgs.append(img)
test_imgs = np.array(test_imgs)
#print(test_imgs, frame_count)

test_images = []
for i in range(test_imgs.shape[0]):
    x = resize(test_imgs[i], preserve_range=True, output_shape=(224,224)).astype(int) # reshaping to 224*224*3
    test_images.append(x)
test_imgs = np.array(test_images)
test_imgs = preprocess_input(test_imgs, mode='tf')
x_test = test_imgs
print(x_test.shape)

predictions = model.predict_classes(test_imgs)

#label prediction for test video 
print("Label predictions Test Video: ",predictions)
length_vid1 = len(predictions)
xtimes = np.arange(0,(length_vid1)/2, 0.5)
ylabels = predictions
data = {}
data["getting into/out of bed"] = []
for i in range(len(xtimes)):
    data["getting into/out of bed"].append([float(xtimes[i]), float(ylabels[i])])
with open("224007215.json", 'w') as file:
    json.dump(data, file)
plt.xlabel("Time in Video (s)")
plt.ylabel("Detection (0 - Not detecting, 1 - Detecting)")
plt.title("Detection for Test Video 1")
plt.plot(np.arange(0,(length_vid1)/2, 0.5), predictions)
plt.savefig("224007215.jpg")
plt.close()
