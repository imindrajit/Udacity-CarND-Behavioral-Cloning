import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, MaxPooling2D, Lambda
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam
import cv2
import json

from model_utils import read_shuffle_split_data, get_training_data_generator, get_validation_data_generator

BATCH_SIZE = 32
LEARNING_RATE = 0.0001


def cnn_model():
    '''NVIDIA Model has been used'''
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(64, 64, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal"))
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", init="he_normal"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", init="he_normal"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164, init="he_normal"))
    model.add(ELU())

    model.add(Dense(100, init="he_normal"))
    model.add(ELU())

    model.add(Dense(50, init="he_normal"))
    model.add(ELU())

    model.add(Dense(10, init="he_normal"))
    model.add(ELU())

    model.add(Dense(1, init="he_normal"))

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')

    return model


if __name__ == "__main__":
    training_data, validation_data = read_shuffle_split_data(split_ratio=0.9)
    model = cnn_model()

    '''Training and Validation generators'''
    training_generator = get_training_data_generator(training_data, batch_size=BATCH_SIZE)
    validation_generator = get_validation_data_generator(validation_data, batch_size=BATCH_SIZE)

    samples_per_epoch = 22000
    validation_samples_per_epoch = 5000

    model.fit_generator(training_generator, samples_per_epoch=samples_per_epoch,
                        nb_epoch=7, validation_data=validation_generator,
                        nb_val_samples=validation_samples_per_epoch)

    print("Saving model weights and configuration file.")

    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())
