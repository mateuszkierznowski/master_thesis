from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys



def conv_3d(self):
    """
    Build a 3D convolutional network, based loosely on C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    """
    # Model.
    model = Sequential()
    model.add(Conv3D(
        32, (3, 3, 3), activation='relu', input_shape=self.input_shape
    ))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(Conv3D(256, (2, 2, 2), activation='relu'))
    model.add(Conv3D(256, (2, 2, 2), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(self.nb_classes, activation='softmax'))

    return model