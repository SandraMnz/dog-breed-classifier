# Dog breed classifier
This is a repo for a project in Deep Learning udacity Nanodegree.

Pytorch library is used.

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project! In this project, you will learn how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app. Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. Your imperfect solution will nonetheless create a fun user experience!

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

## Original Datasets

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)


## CNN to classify dog breed (Building on my own)

Net(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (fc1): Linear(in_features=12544, out_features=500, bias=True)
  
  (fc2): Linear(in_features=500, out_features=133, bias=True)
)

---------

Test Accuracy: 13% (112/836)



## Transfer Learning

Used VGG16 for transfer learning.

Test Accuracy: 85% with 20 epochs.

