# Breast-Caner-Classification
It is a Machine Learning classifier based on IDC dataset to classify a histology image as malignant or benign. (tutorial followed from DataFlair)


In this project in python, I have build a classifier to train on 80% of a breast cancer histology image dataset. Of this, I kept 10% of the data for validation. 
Using Keras, I defined a CNN (Convolutional Neural Network), (CancerNet), and trained it on our images. 
I, then derived a confusion matrix to analyze the performance of the model.

IDC is Invasive Ductal Carcinoma; cancer that develops in a milk duct and invades the fibrous or fatty breast tissue outside the duct; it is the most common form of breast cancer forming 80% of all breast cancer diagnoses. And histology is the study of the microscopic structure of tissues.
(source - Data Flair)

### Prerequisites:

Install some python packages to be able to run this advanced python project. You can do this with pip-

    pip install numpy opencv-python pillow tensorflow keras imutils scikit-learn matplotlib
    
 For Datasets
    
    mkdir datasets
    mkdir datasets\original
    cd breast-cancer-classification\breast-cancer-classification\datasets\original tree
    
config.py:

This holds some configuration we’ll need for building the dataset and training the model. You’ll find this in the cancernet directory.

    import os

    INPUT_DATASET = "datasets/original"

    BASE_PATH = "datasets/idc"
    TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
    VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
    TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    
build_dataset.py:

This will split our dataset into training, validation, and testing sets in the ratio mentioned above- 80% for training (of that, 10% for validation) and 20% for testing. With the ImageDataGenerator from Keras, we will extract batches of images to avoid making space for the entire dataset in memory at once.

 Run the script build_dataset.py:

    py build_dataset.py
    
 cancernet.py:

The network we’ll build will be a CNN (Convolutional Neural Network) and call it CancerNet. This network performs the following operations:

    Use 3×3 CONV filters
    Stack these filters on top of each other
    Perform max-pooling
    Use depthwise separable convolution (more efficient, takes up less memory)
    
 To Sum-up
I learned to build a breast cancer classifier on the IDC dataset (with histology images for Invasive Ductal Carcinoma) and created the network CancerNet for the same. We used Keras to implement the same.
