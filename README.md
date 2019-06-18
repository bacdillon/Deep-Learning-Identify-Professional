# IDENTIFY-PROFESSIONAL
# Problem Description
To recognize professionals by their mode of dressing as humans can observe. 
This is part of mission to train machine learning systems to perceive, understand and act accordingly in any environment they are deployed. 

# Dataset Description
Reason for Creation:
This dataset was created to enable the training of AI systems that can identifiable professionals
Number of images in the dataset

There are 11,000 images contained in the dataset that covers 10 professional examples. 
The dataset is split into :
9000 (900 pictures for each profession) pictures to train the model 
2000 (200 pictures for each profession) pictures to test the performance of the model 
Start with having the same number of images for all the classes

# Example composition
Each professional example in the dataset is contained in a separate folder, with the folder name corresponding to the example label (e.g doctor). 
Size of image 224 * 224 pixel 

# Processing of the image samples
The images in the dataset were obtained from Google Image search. The images were searched and collected based on the 15 countries. 
The dataset was compiled by Moses and John Olafenwa

# Preservation of raw/unprocessed images 
The unprocessed images are preserved. 

# What if don’t have enough images?
Take more photos. Take more videos. Extract frames (images)from videos
Data augmentation / manipulation:
Helps to increase the amount of  relevant data in dataset

# Labelling the Data
For image classification, label the images by having separate folders for each category

# Split data into training / testing :
Remaining 90% will be used for training and validation start with 10% of the images for testing

# Model Training
Train with one model first. If accuracy isn’t up to requirements, try the other ones such as MobileNet V1 or V2 and ResNet50
Carry out a transfer learning example based on Inception-v3 image recognition neural network.

# Inception-v3 consists of two parts:
Feature extraction part with a convolutional neural network.
Classification part with fully-connected and softmax layers.



