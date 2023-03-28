from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Input
from keras import layers
from utils import *

HEIGHT = 224
WIDTH = 224


input_shape = (None, 10, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input
Conv3D(32, (3,3,3), activation='relu')(x)
MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)
model.add(Conv3D(64, (3,3,3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(Conv3D(128, (3,3,3), activation='relu'))
model.add(Conv3D(128, (3,3,3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(Conv3D(256, (2,2,2), activation='relu'))
model.add(Conv3D(256, (2,2,2), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(self.nb_classes, activation='softmax'))