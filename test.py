#tester
# 
# all imports 
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize

model = load_model('bedmodel.h5')
#get test data 
testdata = pd.read_csv('test_map.csv')
test_imgs = []
for each in testdata.Frame_ID:
    img = plt.imread('test_set/' + each + '.jpg')
    test_imgs.append(img)
test_imgs = np.array(test_imgs)
B = testdata.Class 
testclasses = np_utils.to_categorical(B) #one hot encoding

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
print(results, predictions)
