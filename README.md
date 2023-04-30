# ASMOC - Another Simple Modular Object Classifier

## Introduction

ASMOC (Another Simple Modular Object Classifier) is a simple object classifier code that uses cascade xml files to detect objects of interest (OOI) in an input stream. The purpose of this classifier script is to be modular, making it easy to add new OOIs with different properties.<br>This project is intended to be used for learning about computer vision and machine learning.
![Demonstration](demo.gif)

## Input

By default, the program uses the `data/test_cut.mp4` video as input. You can modify this by changing the value of the INPUT_SOURCE constant in the asmoc.py file to for example use the camera feed `(0)` as input.

## Features

- Classify multiple objects in a frame.
- Specify a region of interest (ROI) for each object to improve detection accuracy.
- Use a combination of individual color masks and cascade classifiers for object detection.

## Glossary

- **Object of Interest (OOI)**: An object that the classifier is trained to detect.
- **Region of Interest (ROI)**: A region of the frame that the classifier will search for the OOI in. This is used to improve detection accuracy.
- **Cascade Classifier**: A cascade classifier is a machine learning based approach where a lot of positive and negative images are used to train the classifier. It is then used to detect objects in other images. The algorithm uses the concept of **integral images**, which allows the features used by the classifier to be computed very quickly. For more information, see [this](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) OpenCV tutorial.
- **Color Mask**: A color mask is a binary image that is created by thresholding the input image based on a color range. This is used to detect objects that are of a specific color. For more information, see [this](https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html) OpenCV tutorial.

## Maintainership and Contributing

This project is a work in progress, and is not meant to be used in production environment. It is intended for learning purposes and is subject to change at any time. Better documentation will be added as the project progresses. If you have any suggestions for improvements or would like to contribute to the project, feel free to submit a pull request.