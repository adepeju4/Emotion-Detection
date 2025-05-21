import os
import cv2
import numpy as np
from keras.utils import to_categorical

def process_affectnet(dataset_folder, sub_folders):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # Process training data
    train_dir = os.path.join(dataset_folder, 'Train')
    for label, emotion in enumerate(sub_folders):
        emotion_dir = os.path.join(train_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found - {emotion_dir}")
            continue
            
        print(f"Processing {emotion} images...")
        for image_file in os.listdir(emotion_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(emotion_dir, image_file)
                try:
                    # Use the same preprocessing as FER2013 and CK+
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image - {image_path}")
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (48, 48))
                    X_train.append(image)
                    y_train.append(label)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
    
    # Process test data
    test_dir = os.path.join(dataset_folder, 'Test')
    for label, emotion in enumerate(sub_folders):
        emotion_dir = os.path.join(test_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found - {emotion_dir}")
            continue
            
        print(f"Processing {emotion} test images...")
        for image_file in os.listdir(emotion_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(emotion_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image - {image_path}")
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (48, 48))
                    X_test.append(image)
                    y_test.append(label)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Normalize and reshape exactly as other datasets
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, len(sub_folders))
    y_test = to_categorical(y_test, len(sub_folders))
    
    print("\nDataset processed:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {y_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test 