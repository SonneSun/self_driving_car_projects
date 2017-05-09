# CarND-Behavioral-Cloning-Project3

# ## Overview
#
# 1. Load the training data.
# 2. Preprocess the data.
# 4. Train a convolutional neural network to predict the steering wheel angle
# 5. Save the model to file


import csv
import cv2
import math
import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

print('Modules loaded.')


### Load the Data
images = []
measurements = []

# read sample data
directory = 'sample_data/'
with open('sample_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        steering_center = float(line[3])
        # correction = 0.1
        # steering_left = steering_center + correction
        # steering_right = steering_center - correction

        img_center = cv2.imread(directory + line[0])
        # img_left = cv2.imread(directory + line[1].split(' ')[1])
        # img_right = cv2.imread(directory + line[2].split(' ')[1])

        # images.extend((img_center,img_left,img_right))
        # measurements.extend((steering_center,steering_left,steering_right))
        images.append(img_center)
        measurements.append(steering_center)

# read data of full circle 1
directory = 'training_data/'
with open('training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(steering_center)

# read data of full circle 2
directory = 'training_data4/'
with open('training_data4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(steering_center)

# read data of partial circle
directory = 'training_data3/'
with open('training_data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(steering_center)


# read data of recovery from bountry
directory = 'training_data2/'
with open('training_data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader)
    for line in reader:
        steering_center = float(line[3])
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(steering_center)

# read turn-specific data
directory = 'training_data_circle/'
with open('training_data_circle/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(steering_center)

# read cross-bridge-specific data
directory = 'training_data_bridge/'
with open('training_data_bridge/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(steering_center)



# Data Augmentation
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print('Data prepared.')



### Keras Sequential Model

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(filters=24, kernel_size=(5, 5), activation = 'relu', strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(filters=36, kernel_size=(5, 5), activation = 'relu', strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(filters=48, kernel_size=(5, 5), activation = 'relu', strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#Compile and train the model
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle = True, epochs = 5)
model.save('model.h5')

import gc
gc.collect()
