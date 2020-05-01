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

d = os.path.dirname(os.getcwd())
model = load_model(d+'/bedmodel.h5')
#get test data frames
frame_count = 0
videoFile = 'test1.mp4'
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
"""
test_images = []
for i in range(test_imgs.shape[0]):
    x = resize(test_imgs[i], preserve_range=True, output_shape=(224,224)).astype(int) # reshaping to 256*256*3
    test_images.append(x)
test_imgs = np.array(test_images)
test_imgs = preprocess_input(test_imgs, mode='tf')
x_test = test_imgs
y_test = testclasses
print(x_test.shape, y_test.shape)

results = model.evaluate(x_test, y_test)
predictions = model.predict_classes(test_imgs)
print("Testing Loss: ", results[0])
print("Testing Accuracy: ", results[1])

#you can edit the code below to include your test video's specific frame numbers, or simply test one video at a time

#label prediction for video 1 (frames 0 - 20)
print("Label predictions Test Video 1: ",predictions[0:21])
length_vid1 = len(predictions[0:21])
xtimes = np.arange(0,(length_vid1)/2, 0.5)
ylabels = predictions[0:21]
data = {}
data["getting into/out of bed"] = []
for i in range(len(xtimes)):
    data["getting into/out of bed"].append([float(xtimes[i]), float(ylabels[i])])
with open("timeLabel_1.json", 'w') as file:
    json.dump(data, file)
plt.xlabel("Time in Video (s)")
plt.ylabel("Detection (0 - Not detecting, 1 - Detecting)")
plt.title("Detection for Test Video 1")
plt.plot(np.arange(0,(length_vid1)/2, 0.5), predictions[0:21])
plt.savefig("test-vid-1.jpg")
plt.clf()

#label prediction for video 2 (frames 21 - 33)
print("Label predictions Test Video 2: ",predictions[21:34])
length_vid2 = len(predictions[21:34])
xtimes = np.arange(0,(length_vid2)/2, 0.5)
ylabels = predictions[21:34]
data = {}
data["getting into/out of bed"] = []
for i in range(len(xtimes)):
    data["getting into/out of bed"].append([float(xtimes[i]), float(ylabels[i])])
with open("timeLabel_2.json", 'w') as file:
    json.dump(data, file)
plt.xlabel("Time in Video (s)")
plt.ylabel("Detection (0 - Not detecting, 1 - Detecting)")
plt.title("Detection for Test Video 2")
plt.plot(np.arange(0,(length_vid2)/2, 0.5), predictions[21:34])
x1,x2,y1,y2 = plt.axis()
plt.savefig("test-vid-2.jpg")
plt.clf()

#label prediction for video 3 (frames 34 - 51)
print("Label predictions Test Video 3: ",predictions[34:52])
length_vid3 = len(predictions[34:52])
xtimes = np.arange(0,(length_vid3)/2, 0.5)
ylabels = predictions[34:52]
data = {}
data["getting into/out of bed"] = []
for i in range(len(xtimes)):
    data["getting into/out of bed"].append([float(xtimes[i]), float(ylabels[i])])
with open("timeLabel_3.json", 'w') as file:
    json.dump(data, file)
plt.xlabel("Time in Video (s)")
plt.ylabel("Detection (0 - Not detecting, 1 - Detecting)")
plt.title("Detection for Test Video 3")
plt.plot(np.arange(0,(length_vid3)/2, 0.5), predictions[34:52])
x1,x2,y3,y4 = plt.axis()
plt.axis((x1, x2, y1, y2))
plt.savefig("test-vid-3.jpg")
plt.clf()

#label prediction for video 4 (frames 52 - 74)
print("Label predictions Test Video 4: ",predictions[52:75])
length_vid4 = len(predictions[52:75])
plt.xlabel("Time in Video (s)")
plt.ylabel("Detection (0 - Not detecting, 1 - Detecting)")
plt.title("Detection for Test Video 4")
xtimes = np.arange(0,(length_vid4)/2, 0.5)
ylabels = predictions[52:75]
data = {}
data["getting into/out of bed"] = []
for i in range(len(xtimes)):
    data["getting into/out of bed"].append([float(xtimes[i]), float(ylabels[i])])
with open("timeLabel_4.json", 'w') as file:
    json.dump(data, file)
plt.plot(np.arange(0,(length_vid4)/2, 0.5), predictions[52:75])
x1,x2,y3,y4 = plt.axis()
plt.axis((x1, x2, y1, y2))
plt.savefig("test-vid-4.jpg")
plt.clf()

#label prediction for video 5 (frames 75 - 103)
print("Label predictions Test Video 5: ",predictions[75:103])
length_vid5 = len(predictions[75:103])
xtimes = np.arange(0,(length_vid5)/2, 0.5)
ylabels = predictions[75:103]
data = {}
data["getting into/out of bed"] = []
for i in range(len(xtimes)):
    data["getting into/out of bed"].append([float(xtimes[i]), float(ylabels[i])])
with open("timeLabel_5.json", 'w') as file:
    json.dump(data, file)
plt.xlabel("Time in Video (s)")
plt.ylabel("Detection (0 - Not detecting, 1 - Detecting)")
plt.title("Detection for Test Video 5")
plt.plot(np.arange(0,(length_vid5)/2, 0.5), predictions[75:103])
x1,x2,y3,y4 = plt.axis()
plt.axis((x1, x2, y1, y2))
plt.savefig("test-vid-5.jpg")
plt.close()
"""
