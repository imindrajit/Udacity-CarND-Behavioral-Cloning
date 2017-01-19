# Udacity-CarND-Behavioral-Cloning


# About

This project is part of Udacity's Self-Driven Car Nanodegree Program. The car has to drive through by itself on a track from a video game. Convolutional Neural Networks have been used for predicting the steering angle for the car.


# Data Exploration

Udacity data has been used in this project. There is a file **data_exploration.ipynb** which  shows information about the data provided. Also, a sample output of an input through the augmentation pipeline is provided in the same file. 


# Training and Validation Data Split

As evident from the data exploration file, most of the steering angles are in and around 0, so, to reduce high bias :-

1) 25% of the data is randomly dropped from training in which steering angle is 0.

2) 75% of the data where steering angle is 0 and rest of the data where steering angle is not 0 are mixed together.

3) 90% of the mixed data is used for training and rest 10% for validation. 

4) There is no need for test data as the track on which the car runs is itself a test data.


# Data Preprocessing

There are around 8000 unique rows in **driving_log.csv**. Each row has center, left and right images.

1) In training generator, any row from the training data is randomly picked.

2) Then, either of the left, right or center images are also selected randomly. Also, the steering angle associated with the images.

3) If selected image is that of center then, the original steering angle remains same.

4) If selected image is that of left then, a fixed angle of 0.17 is added to the original steering angle.

5) If selected image is that of right then,  a fixed angle of -0.17 is added to the original steering angle.


# Augmentation Techniques

   Shearing
  
      * Shearing is applied randomly on 90% of the training images. Rest 10% are left as it.
      
      * Bend the image randomly and also change the corresponding steering angle associated with it.
   
   Crop Image
   
      * Informations like horizon in background and bonnet of car are not useful for predicting the steering angle. So, we will remove them.
      
      * Top 25% of the image and bottom 25 units are cropped away.
   
   Flip
      
      * Randomly flip 50% of the training image so that we can have more generalized images.
      
      * If flipping condition is true, then flip the image and also steering_angle = -1 * original_steering_angle.
   
   Brightness
   
      * Change the brightness of the images randomly so that the model gets acclimatized to different light conditions.
      
      * The image is converted from RGB to HSV color channel.
      
      * The V channel is then randomly multiplied by some value.
      
      * The new image is again converted back to RGB.
   
   Resize
   
      * The image obtained after passing through the above steps is reshaped to (64, 64, 3).

Data is augmented/generated on the fly using python generators. So, the model effectively sees new images everytime.

I would like to thank [Vivek](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jwzy6grgx) and [Kaspar Sakmann](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.8xghuqf53) for their wonderful blogposts which really helped me out a lot here.


# Model

I have used [NVIDIA](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)  model as convolutional neural network in this project.
  
  Input Image
     
     * Size -> (64, 64, 3)
  
  Layer 1
  
     * Lamba layer which normalizes all pixel values between [-1, 1]. Normalization helps the model to converge faster.
  
  Layer 2
      
     * 24 - 5 X 5 filters. ELU as non-linear function. MaxPool of 2 X 2 and stride of 1 X 1.
  
  Layer 3
      
     * 36 - 5 X 5 filters. ELU as non-linear function. MaxPool of 2 X 2 and stride of 1 X 1.
   
  Layer 4
      
     * 48 - 5 X 5 filters. ELU as non-linear function. Dropout of 0.5. MaxPool of 2 X 2 and stride of 1 X 1.
  
  Layer 5
      
     * 64 - 3 X 3 filters. ELU as non-linear function. MaxPool of 2 X 2 and stride of 1 X 1.
  
  Layer 6
  
     * 64 - 3 X 3 filters. ELU as non-linear function. MaxPool of 2 X 2 and stride of 1 X 1.
     
  Layer 7
  
     * Flatten and then Fully Connected Layer of 1164. ELU as non-linear function.
  
  Layer 8
  
     * Fully Connected Layer of 100. ELU as non-linear function.
  
  Layer 9
  
     * Fully Connected Layer of 50. ELU as non-linear function.
  
  Layer 10
  
     * Fully Connected Layer of 10. ELU as non-linear function.
  
  Layer 11
  
     * Final Output Layer of 1.


# Training Method

   * Batch Size = 32
   
   * Learning Rate = 0.0001
   
   * Optimizer -> Adam
   
   * Training Samples per epoch = 25000. All samples are generated on the fly using fit_generator function of keras.
   
   * Validation Samples per epoch = 5000. All samples are generated on the fly using fit_generator function of keras.
   
   * Number of epochs = 5
   
 
# Video Links
 
1) [Track 1](https://www.youtube.com/watch?v=QJXFayJB92E)
 
2) [Track 2](https://www.youtube.com/watch?v=HbOW_BBhFI4&t=84s)
 
 
# Future Scope for Improvement
 
 1) If graphics quality is increased then the model struggles it's way through the tracks. The reason being that low quality images were used for training. So, ideally simulator should be used to generate high quality images and train the model on those images.
 
 2) Throttle is constant at 0.3 for now but Adaptive throttling should be done so as to help the car to run around the track in more real life like scenario.
 
 3) Weather conditions may change. So, training for different weather conditions need to taken care of so that the model becomes much more generalized.
