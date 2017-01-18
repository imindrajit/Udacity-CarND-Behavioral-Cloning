import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
import os
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2
from scipy.stats import bernoulli

DATA_FOLDER = "data/"


def read_shuffle_split_data(split_ratio=0.8):
    '''Read Data'''
    df = pd.read_csv('data/driving_log.csv')

    '''Use only randomly shuffled 75% data for which steering angle is 0 in training'''
    data_zero_steering = df[df['steering'].isin([0])].reset_index()
    data_zero_steering_shuffled = data_zero_steering.sample(frac=1).reset_index(drop=True)
    num_rows = data_zero_steering_shuffled.shape[0] * 0.75
    training_zero_steering = data_zero_steering_shuffled.loc[:num_rows - 1]

    training_non_zero_steering = df[~df['steering'].isin([0])].reset_index()

    '''Combine zero steering angle and non-zero steering angle data'''
    all_training = training_zero_steering.append(training_non_zero_steering, ignore_index=True)
    '''Shuffle Data'''
    new_df = all_training.sample(frac=1).reset_index(drop=True)

    '''Training Validation Split'''
    num_rows = new_df.shape[0] * split_ratio

    '''Split the data into split_ratio:1 for training and validation sets'''
    training_data = new_df.loc[:num_rows - 1]
    validation_data = new_df.loc[num_rows:]

    return training_data, validation_data


def get_image_steering_angle_raw(data):
    '''
        1) For every data left, right or center image is selected randomly based on the output of random_val.
        2) If image type center is selected then, steering angle doesn't need to change.
        3) If image type left is selected then, steering angle is incremented by 0.17 so that the car points towards center.
        4) If image type right is selected then, steering angle is decremented by 0.17 so that the car points towards center.
    '''
    batch_images = []
    batch_steering = []

    for ind, row in data.iterrows():
        angle_shift, img_loc = 0, None
        steering_angle = row['steering']
        random_val = np.random.randint(3)
        if random_val == 0:
            img_loc = row['center']
            angle_shift = 0
        elif random_val == 1:
            img_loc = row['left']
            angle_shift = 0.17
        elif random_val == 2:
            img_loc = row['right']
            angle_shift = -0.17
        steering_angle += angle_shift
        raw_img = mpimg.imread("{0}{1}".format(DATA_FOLDER, img_loc.strip()))
        batch_images.append(raw_img)
        batch_steering.append(steering_angle)
    return np.array(batch_images), np.array(batch_steering)


def resize_image(image):
    '''Resize image to (64, 64, 3)'''
    final_shape = (64, 64)
    image_final = cv2.resize(image, final_shape, interpolation=cv2.INTER_AREA)
    return image_final


def change_brightness(image):
    '''Convert image from RGB to HSV and randomly change the V component. Then convert back to RGB'''
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright_rand = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * bright_rand
    image_final = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return image_final


def random_shear(image, steering_angle, shear_range=200):
    '''Randomly shear the image and the corresponding steering angle is changed'''
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def crop_image(image):
    '''
        1) Cropping is done only on Y axis i.e, vertically.
        2) Remove 25% of the topmost part of the image which contains horizon and also the bottom 25 units
           as which is part of the bonnet.
    '''
    shape = image.shape
    img = image[shape[0] // 4:shape[0] - 25][0:shape[1]]
    return img


def random_flip(image, steering_angle):
    '''Flip the image if rand_val is 1. So, steering_angle becomes negative of the original value'''
    rand_val = np.random.randint(1)
    if rand_val:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    return image, steering_angle


def transform_image(image, steering_angle):
    '''
        1) Apply shear to 90% of the data and the other 10% is left as it is.
        2) Crop Image.
        3) Randomly Flip the image with 50% probability.
        4) Change brightness of the image randomly.
        5) Resize image to (64, 64, 3)
    '''
    rand_prob = bernoulli.rvs(0.9)
    if rand_prob:
        image, steering_angle = random_shear(image, steering_angle)
    img = crop_image(image)
    img, steer_angle = random_flip(img, steering_angle)
    img = change_brightness(img)
    img = resize_image(img)
    return img, steer_angle


def preprocess_validation_data(raw_images, raw_angles):
    batch_images = []
    batch_steering = []

    for index, each_img in enumerate(raw_images):
        steering_angle = raw_angles[index]
        cropped_image = crop_image(each_img)
        resized_image = resize_image(cropped_image)
        batch_images.append(resized_image)
        batch_steering.append(steering_angle)
    return np.array(batch_images), np.array(batch_steering)


def preprocess_augment_data(raw_images, raw_angles):
    batch_images = []
    batch_steering = []

    for index, each_img in enumerate(raw_images):
        str_angle = raw_angles[index]
        augmented_image, steering_angle = transform_image(each_img, str_angle)
        batch_images.append(augmented_image)
        batch_steering.append(steering_angle)
    return np.array(batch_images), np.array(batch_steering)


def get_training_data_generator(data_df, batch_size):
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_angles = np.zeros(batch_size)

    while True:
        for i in range(batch_size):
            '''Shuffle and randomly select any data frame'''
            new_df = data_df.sample(frac=1).reset_index(drop=True)
            idx = np.random.randint(len(new_df))

            data_row = new_df.iloc[[idx]].reset_index()
            '''Get raw image and steering angle'''
            raw_images, raw_angles = get_image_steering_angle_raw(data_row)
            '''Apply augmentation on the image'''
            img, angle = preprocess_augment_data(raw_images, raw_angles)

            batch_images[i] = img
            batch_angles[i] = angle

        yield batch_images, batch_angles


def get_validation_data_generator(data_df, batch_size):
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_angles = np.zeros(batch_size)

    while True:
        for i in range(batch_size):
            '''Shuffle and randomly select any data frame'''
            new_df = data_df.sample(frac=1).reset_index(drop=True)
            idx = np.random.randint(len(new_df))

            data_row = new_df.iloc[[idx]].reset_index()
            '''Get raw image and steering angle'''
            raw_images, raw_angles = get_image_steering_angle_raw(data_row)
            img, angle = preprocess_validation_data(raw_images, raw_angles)

            batch_images[i] = img
            batch_angles[i] = angle

        yield batch_images, batch_angles
