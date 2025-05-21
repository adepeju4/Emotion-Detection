import os
import cv2
import numpy as np
import tensorflow as tf
    
def custom_train_test_split(X, y, test_size=0.2):

    n_samples = len(X)
    
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def process_ck_plus(dataset_folder, sub_folders):
   
    images = []
    labels = []
    
    
    print("Processing CK+ dataset...")
    
    for sub_folder in sub_folders:
        label = sub_folders.index(sub_folder)
        path = os.path.join(dataset_folder, sub_folder)
        
        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            print(f"Processing {sub_folder}: {image_path}")
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (48, 48)) 
            
            images.append(image)
            labels.append(label)
    

    X = np.array(images)
    y = np.array(labels)
    
    X = X.astype('float32') / 255.0
    X = X.reshape(X.shape[0], 48, 48, 1)
    y = tf.keras.utils.to_categorical(y, num_classes=len(sub_folders))
    
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
    
    print(f"Dataset processed:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {y_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

