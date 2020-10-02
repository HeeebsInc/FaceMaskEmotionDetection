import pickle
import cv2
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess

def get_image_value(path, dim, bw, model_type): 
    '''This function will read an image and convert to a specified version and resize depending on which algorithm is being used.  If edge is specified as true, it will pass the img array to get_edged which returns a filtered version of the img'''
    img = image.load_img(path, target_size = dim)
    img = image.img_to_array(img)
    if bw == True: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(img.shape[0], img.shape[1],1)
    if model_type == 'mobilenet': 
        img = mobile_preprocess(img)
        return img
    elif model_type == 'vgg16': 
        img = vgg16_preprocess(img) 
        return img
    return img/255


def get_emotion_classes(class_type, max_values = 6000): 
    angry_paths = [f'../EmotionDataset/{class_type}/angry/{i}' for i in os.listdir(f'../EmotionDataset/{class_type}/angry')][:max_values]
    angry_labels = [0 for i in range(len(angry_paths))]
    
    happy_paths = [f'../EmotionDataset/{class_type}/happy/{i}' for i in os.listdir(f'../EmotionDataset/{class_type}/happy')][:max_values] 
    happy_labels = [1 for i in range(len(happy_paths))]
    
    neutral_paths = [f'../EmotionDataset/{class_type}/neutral/{i}' for i in os.listdir(f'../EmotionDataset/{class_type}/neutral')][:max_values] 
    
    neutral_labels = [2 for i in range(len(neutral_paths))]
    
    sad_paths = [f'../EmotionDataset/{class_type}/sad/{i}' for i in os.listdir(f'../EmotionDataset/{class_type}/sad')][:max_values] 
    sad_labels = [3 for i in range(len(sad_paths))]
    
#     disgust_paths = [f'../EmotionDataset/{class_type}/disgust/{i}' for i in os.listdir(f'../EmotionDataset/{class_type}/disgust')]
#     disgust_labels = [4 for i in range(len(disgust_paths))]

#     fear_paths = [f'../EmotionDataset/{class_type}/fear/{i}' for i in os.listdir(f'../EmotionDataset/{class_type}/fear')]
#     fear_labels = [5 for i in range(len(fear_paths))]
    
#     surprise_paths = [f'../EmotionDataset/{class_type}/surprise/{i}' for i in os.listdir(f'../EmotionDataset/{class_type}/surprise')] 
#     surprise_labels = [6 for i in range(len(surprise_paths))]
    
    labels = np.array(angry_labels + happy_labels + \
                        neutral_labels + sad_labels)
    
    print(f'{class_type.upper()} Value Count')
    print(pd.Series(labels).value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    labels = to_categorical(labels)
    paths = np.array(angry_paths + happy_paths + neutral_paths + \
                          sad_paths)
    paths, labels = sk_shuffle(paths, labels)
    return paths, labels
#0: angry  1: disgust  2: fear  3: happy  4: neutral  5: sad  6: surprise
def get_emotion_splits(dim, model_type = 'Mobilenet', bw = False): 
    
    #Train
    
    train_paths, train_labels = get_emotion_classes('train')
    test_paths, test_labels = get_emotion_classes('test')
    
    train_images = np.array([get_image_value(i, dim, bw, model_type) for i in train_paths])
    test_images = np.array([get_image_value(i, dim, bw, model_type) for i in test_paths])
    if model_type == 'Mobilenet' and bw == True: 
        train_images = np.stack((train_images,)*3, axis =-1)
        test_images = np.stack((test_images,)*3, axis = -1)
    
    
    return train_images, test_images, train_labels, test_labels 