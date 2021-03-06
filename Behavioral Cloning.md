# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_plot.png "Model Visualization"
[image2]: ./images/cent.png "Center drive"
[image3]: ./images/left.png "Left Recovery Image"
[image4]: ./images/right.png "Right Recovery Image"
[image5]: ./images/orig.png "Original Image"
[image6]: ./images/flip.png "Flipped Image"
[image7]: ./images/muddy_road.png "Turn near Muddy road"
[image8]: ./images/S-layer.png "Robust S layer"
[image9]: ./images/loss_plot.png "Loss plot"
[image10]: ./images/run1.mp4 "Autonomous run"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 8 and 32 (model.py lines 102-117) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 103). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 109,113 and 116). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet as I thought this model might be appropriate because of its simlicity and ability to model nolinear systems.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layers and adding more training data. I was able to solve the occasional instability due to surrounding terrain by preprocessing the training and validation images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and went off the road. To improve the driving behavior in these cases, I included more convolution layers, gathered more training data and performed data preprocessing.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
![alt text][image10]

#### 2. Final Model Architecture

The final model architecture (model.py lines 102-117) consisted of a convolution neural network with the following layers and layer sizes.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 1)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 1)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 66, 316, 8)        208       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 33, 158, 8)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 154, 16)       3216      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 77, 16)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 77, 16)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 73, 32)        12832     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 36, 32)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 5760)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 5760)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 150)               864150    
_________________________________________________________________
activation_1 (Activation)    (None, 150)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 150)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 151       
=================================================================
Total params: 880,557
Trainable params: 880,557
Non-trainable params: 0
_________________________________________________________________

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]


Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]



After the collection process, I had 18828 number of data points. I then preprocessed this data by converting the images into HSV colour space. The network is designed to process only the Saturation layer of image. I then increased the contrast of the HSV image to make the saturation layer robust against muddy terrains.(Code lines: 26-31). The model also performs a cropping operation on the image to avoid unwanted data.

![alt text][image7]
![alt text][image8]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the below validation and training loss plot. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image9]

