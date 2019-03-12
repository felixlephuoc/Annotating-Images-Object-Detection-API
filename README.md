
# Table of contents
I. [Overview](#overview)
1. [Problem Description](#problem_description)
2. [Tensorflow Object Detection API](#object_detection_api)
3. [Annotating Image](#annotating_image)
    
II. [Preliminaries](#preliminaries)


# I. Overview <a name="overview"></a>
## 1.Problem Description <a name="problem_description"></a>

Computer vision has made great leaps forward in recent years because of deep learning, thus granting computers a higher grade in understanding visual scenes. The potentialities of deep learning in vision tasks are great: allowing a computer to visually perceive and understand its surrounding is a capability that opens the door to new artificial intelligence applications in mobility, manufacturing, healthcare and many other human-machine interaction contexts. For instance, self-driving cars can detect if an appearing obstacle is a pedestrian, an animal or another vehicle from the camera mounted on it and decide the correct course of action. Meanwhile, a robot with "seeing" capability can recognize surrounding objects and successfully interact with them.

This project presents a quick and handy way to make computer understand images and videos collected from the Internet or directly taken from webcam. The goal of this project is to find the exact location and the type of the objects in an image.

## 2. Tensorflow Object Detection API <a name="object_detection_api"></a>

In order to achieve such classification and localization, we wil leverage the TensorFlow object detection  API. This is a Google's open source frame work built on top of TensorFlow which is focused on finding objects in images (estimating the chance that an object is in this position) and their bounding boxes. The framework offers some useful functions and these five pre-trained different models:
    * Single Shot Multibox Detector (SSD) with MobileNets
    * SSD with Interception V2
    * Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101
    * Faster R-CNN with Resnet 101
    * Faster R-CNN with Interception Resnet v2
The model are in growing order of precision in detection and slower speed of execution of the detection process. **MobileNets, Inception** and **Resnet** refer to different types of Convolution Neural Network (CNN) architecture. **Single Shot Multibox Detector (SSD), Region Based Fully convolutional networks (R-FCN)** and **Faster Region-based convolutional neural network (Faster R-CNN)** are instead the different models to detect multiple objects in images. A source of reference with more details about these detection models can be found [here](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9).

## 3. Annotating Image <a name="annotating_image"></a>

* Given such a powerful tool made available by TensorFlow, our plan is to leverage its API by creating a class you can use for annotating images both visually and in an external file. By annotating we mean the following:
    - Pointing out the objects in an image 
    - Reporting the level of confidence in the object recognition (only consider objects above a minimum probability threshold, which is set at 0.25)
    - Outputting the coordinates of two opposite vertices of the bounding box for each image
    - Saving all such information in a text file in JSON format
    - Visually representing the bounding box on the original image, if required





* In order to achieve such objectives, we need to:
    1. Download one of pre-trained models (available in .pb format - protobuf) and make it available in-memory as a TensorFlow session
    2. Reformulate the helper code provided by TensorFlow in order to make it easier to load labels, categories and visualization tools
    3. Prepare a simple script sto demonstrate its uage with single images, videos and videos captured from a webcam

Let's first start by seeting up an environment suitable for the project.

# II. Preliminaries <a name="priliminaries"></a>
## 1. Setting up working environment
For this project, a separated environment named *TensorFlow_api* is created using Anaconda *conda*. The commands in terminal are as follows: 



```shell
conda create -n TensorFlow_api_2 python=3.5 numpy pillow
source activate TensorFlow_api_2
```



After activating the environment, we install the necessary packages using *conda install*:


```shell
conda install TensorFlow-gpu
conda install -c menpo opencv
conda install -c conda-forge imageio
conda install tqdm
conda install -c conda-forge moviepy
```


```python
In case 
```

# III. Provision of the project code

Steps by steps showing code of *object_detection.py*

# IV. Result and applicatitons
1. Annotating images with file_pipeline.py
2. Annotating video with video_pipeline.py
3. Annotating screenshot captured by webcam with webcam_pipeline.py
4. Real-time webcam detection
