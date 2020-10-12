# Mask-Emotion Detection

**AI powered system for detecting whether a person is wearing a mask and if not, their facial expression** 

## Contributers
- Samuel Mohebban (B.A. Psychology - George Washington University)
    - Samuel.MohebbanJob@gmail.com
    - [LinkedIn](https://www.linkedin.com/in/samuel-mohebban-b50732139/)
    - [Medium](https://medium.com/@HeeebsInc)

## Business Problem
- In the age of COVID-19, mask protection has been a vital instrument for stopping spread of the virus.  However, there has been much debate over whether people should be forced to wear one.  
- Many businesses require that all customers wear a face mask, and have been found to enforce these rules by not allowing people to enter if they are found not wearing a face covering.
- Even though these rules are clear, many people around the United States have refused to follow these rules, and have causes disruptions at local business for being told they cannot enter without a mask

## Solution
- This project is meant to fix this issue by detecting whether a person is wearing a mask, and if they are not, their facial expression will be read to determine if they are disgruntled. 
- This detection will use a Convolutional Neural Network to read each frame of a video and make these detections

## Requirements
- `keras` (PlaidML backend --> GPU: RX 580 8GB)
- `numpy`
- `opencv`
- `matplotlib`

## Data
- Two datasets were used to train two neural networks
1. Face Mask: ~12K Images Dataset ([Kaggle](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset?select=Face+Mask+Dataset))
2. Emotion: ~30K Images Dataset ([Kaggle](https://www.kaggle.com/msambare/fer2013))

## Modeling
- For each model, early stopping was applied to prevent the model from overfitting
- The data for the models can be retrieved using the functions within the [Functions.py](PyFunctions/Functions.py) file
- Visualizations functions for the confusion matrix, loss/accuracy, and ROC curve can be found within the [Viz.py](PyFunctions/Viz.py) file 

**1. Face Mask Detection: Mobilenet**
- [Notebook](MobilenetMasks.ipynb)
- Images were resized to (224, 224)
- Weights for imagenet were applied as well as sigmoid activation for the output layer, and binary crossentropy for the loss function
- Mobilenet was trained on the mask dataset
- Augmentation was applied to the training to ensure the model is able to generalize predictions to unknown data 


<p align="center" width="100%">
    <img width="50%" src="Images/Mobilenet_Loss_Acc.png"> 
    <img width="50%" src="Images/Mobilenet_ROC_F1.png"> 
</p>

**2. Emotion Detection: Convolutional Neural Network** 
- [Notebook](NormalEmotions.ipynb)
- Images were resized to (48,48)
- Softmax activation and categorical crossentropy were applied


## LIME Feature Extraction
- [Notebook](LimeFE.ipynb)
- In this section of the notebook, I use LIME- a python package that can be used for feature extraction of black box models
- Below, the areas that are green are those that the algorithm deems "important" for making a prediction
- This technique is useful because it allowed me to understand what the neural network is basing its predictions off of

#### Mask Detection: Mobilenet

#### Emotion Detection: Convolutional Neural Network

## Deployment
- [Notebook](FaceDetector.ipynb)
- In this section, the models were applied to live video
- The steps for applying this to live video were as follows:\
    1. Use [Haar feature based cascade classifier](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) to locate the coordinates of a face within a frame
    2. Extract the ROI of the face using the coordinates given by the classifier
    3. Make two copies of the ROI, one for the mask model and another for the emotion model
   44. Resize each copy to the correspoding dimensions used within the models 
    5. Start by making a mask prediction
        - If the model detects there is a mask, it will stop predicting and show a green box
        - If the model does not detect the mask, the algorithm will move onto the emotion model
    - Below you can see a .gif of how it works on my local machine

## Future Directions