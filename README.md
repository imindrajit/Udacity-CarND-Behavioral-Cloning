# Udacity-CarND-Behavioral-Cloning

# About

This project is part of Udacity's Self-Driven Car Nanodegree Program. The car has to drive through by itself on a track from a video game. Convolutional Neural Networks have been used for predicting the steering angle for the car.

# Data Exploration

Udacity data has been used in this project. There is a file **data_exploration.ipynb** which  shows information about the data provided. Also, a sample output of an input through the augmentation pipeline is provided in the same file. 

# Training and Validation data Split

As evident from the data exploration file, most of the steering angles are in and around 0, so, to reduce high bias :-

1) 25% of the data is randomly dropped from training in which steering angle is 0.

2) 75% of the data where steering angle is 0 and rest of the data where steering angle is not 0 are mixed together.

3) 90% of the mixed data is used for training and rest 10% for validation. 

4) There is no need for test data as the track on which the car runs is itself a test data.


# Data Preprocessing

There are around 8000 unique rows in **driving_log.csv**. Each row has center, left and right images.



# Augmentation Techniques

Finally, I would like to thank [Vivek](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jwzy6grgx) and [Kaspar Sakmann](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.8xghuqf53) for their wonderful blogposts which really helped me out a lot.
