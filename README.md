# Project: Distracted driver detection and classification using deep convolutional neural networks

## Description: 
This project aims to develop a machine learning system that can detect and classify different distracted states of car drivers. The main approach is to apply deep convolutional neural networks (CNNs). We will explore and experiment various CNN architectures, leveraged pre-trained networks (learning transfer), psuedo labelling, and potentially an emsenbles of several models to find the best classification. Results of this project may be used to further research and applied to as a part of an on-car online monitoring system where computer will decide to take-over control of the car if the driver is distracted and poses a potential accident.

## Data
Data is a collection of 10 different states of drivers containing one safe driving and 9 other distracted modes. The dataset is provided by State Farm through Kaggle which can be downloaded from [here](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data). Data exploration and preprocessing will be described in details in subsequent sections.

## Analysis
Data exploration, preprocessing and analysis will be conducted in great details to gain as much information about the dataset as possible. All steps of a machine learning pipeline are included and a summary is provided at the end of each section.

## Tools

The project utilizes the following dependencies:

- Python 3.5: Tensorflow, Keras, Numpy, Scipy
- NVIDIA Geforce GTX960 GPU, CUDA, CuDNN

## Data Preprocessing, Modelling and Traning

Details are documented in this [notebook](https://github.com/tnaduc/Distracted_Driver_Detect_Classify/blob/master/Distracted_Driver_Detection_Classification.ipynb).
