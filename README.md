
# Table of contents
1. [Introduction](#introduction)
2. [Setting up environment](#preliminaries)


#Problem Description <a name="introduction"></a>

- What does this project do?
Computer vision has a lot of application for eg... -> This project present a quick, easy way to make computer understand images and videos collected from the Internet or directly taken from computer's webcam. The goal of this project is to find the exact location and the type of the objects in an image


* In order to achieve such classification and localization, we wil leverage the TensorFlow object detection  API. This is a Google's open source frame work built on top of TensorFlow which is focused on finding objects in images (estimating the chance that an object is in this position) and their bounding boxes. The framework offers some useful functions and these five pre-trained different models:
    * Single Shot Multibox Detector (SSD) with MobileNets
    * SSD with Interception V2
    * Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101
    * Faster R-CNN with Resnet 101
    * Faster R-CNN with Interception Resnet v2
The model are in growing order of precision in detection and slower speed of execution of the detection process. 

* Given such a powerful tool made available by TensorFlow, our plan is to leverage its API by creating a class you can use for annotating images both visually and in an external file. By annotating we mean the following:
    * Pointing out the objects in an image 
    * Reporting the level of confidence in the object recognition (only consider objects above a minimum probability threshold, which is set at 0.25)
    * Outputting the coordinates of two opposite vertices of the bounding box for each image.
    * Saving all such information in a text file in JSON format
    * Visually representing the bounding box on the original image, if required.

* The project are divided into 3 steps:
    1. Download one of pre-trained models (available in .pb format - protobuf) and make it available in-memory as a TensorFlow session.
    2. Reformulate the helper code provided by TensorFlow in order to make it easier to load labels, categories and visualization tools
    3. Prepare a simple script sto demonstrate its uage with single images, videos and videos captured from a webcam.



## Preliminaries <a name="preliminaries"></a>
- Download the Tensorflow object detection code.
- Setting up  an environment: conda install...
- Protobuf compilation


## Provision of the project code

Steps by steps showing code of *object_detection.py*

## Result and applicatitons
1. Annotating images with file_pipeline.py
2. Annotating video with video_pipeline.py
3. Annotating screenshot captured by webcam with webcam_pipeline.py
4. Real-time webcam detection


