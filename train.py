#trainer
# 
# all imports 
import keras
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize

data = pd.read_csv('train_map.csv')
#print(data.head())
imgs = []
for each in data.Frame_ID:
    img = plt.imread('training-set/' + each + '.jpg')
    imgs.append(img)
imgs = np.array(imgs)
print(imgs.shape)
C = data.Class 
classes = np_utils.to_categorical(C) #one hot encoding
print(imgs[0].shape)
images = []
for i in range(imgs.shape[0]):
    x = resize(imgs[i], preserve_range=True, output_shape=(224,224)).astype(int) # reshaping to 256*256*3
    images.append(x)
imgs = np.array(images)
imgs = preprocess_input(imgs, mode='tf')

x_train, x_valid, y_train, y_valid = train_test_split(imgs, classes, test_size=0.25, random_state=42)
#normalize pixel values
x_train = x_train / 255
x_valid = x_valid / 255
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

#building model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

learning = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
model.save('bedmodel.h5')