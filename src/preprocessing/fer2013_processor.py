import os
import cv2
import numpy as np
from keras.utils import to_categorical

def preprocess_face_for_emotion(face_img, target_size=(48, 48)):
    """
    Preprocess a face image for emotion detection
    
    Args:
        face_img: The cropped face image
        target_size: Target size for the model (default: 48x48)
        
    Returns:
        Preprocessed face ready for the model
    """
    # Convert to grayscale if needed
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_img
        
    # Resize to target size
    resized_face = cv2.resize(gray_face, target_size)
    
    # Normalize to [0, 1] range
    normalized_face = resized_face.astype('float32') / 255.0
    
    # Expand dimensions for model
    if len(normalized_face.shape) == 2:
        normalized_face = np.expand_dims(normalized_face, axis=-1)
    
    return normalized_face

def process_fer2013(dataset_folder, sub_folders):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # Process training data
    train_dir = os.path.join(dataset_folder, 'train')
    total_processed = 0
    
    for label, emotion in enumerate(sub_folders):
        emotion_dir = os.path.join(train_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found - {emotion_dir}")
            continue
            
        image_files = [f for f in os.listdir(emotion_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        
        print(f"\nProcessing {emotion} training images ({total_images} files)...")
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(emotion_dir, image_file)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image - {image_path}")
                    continue
                
                # Apply comprehensive preprocessing
                processed_face = preprocess_face_for_emotion(image)
                X_train.append(processed_face)
                y_train.append(label)
                total_processed += 1
                
                if idx % 100 == 0:  # Progress update every 100 images
                    print(f"Processed {idx}/{total_images} {emotion} images...")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    print(f"\nCompleted training set processing. Total images: {total_processed}")
    
    # Process test data
    test_dir = os.path.join(dataset_folder, 'test')
    total_test_processed = 0
    
    for label, emotion in enumerate(sub_folders):
        emotion_dir = os.path.join(test_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory not found - {emotion_dir}")
            continue
            
        image_files = [f for f in os.listdir(emotion_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        
        print(f"\nProcessing {emotion} test images ({total_images} files)...")
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(emotion_dir, image_file)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image - {image_path}")
                    continue
                
                # Apply comprehensive preprocessing
                processed_face = preprocess_face_for_emotion(image)
                X_test.append(processed_face)
                y_test.append(label)
                total_test_processed += 1
                
                if idx % 100 == 0:  # Progress update every 100 images
                    print(f"Processed {idx}/{total_images} {emotion} test images...")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    print(f"\nCompleted test set processing. Total images: {total_test_processed}")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, len(sub_folders))
    y_test = to_categorical(y_test, len(sub_folders))
    
    print("\nDataset processed:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {y_train.shape[1]}")
    print(f"Memory usage: Train={X_train.nbytes / 1e6:.1f}MB, Test={X_test.nbytes / 1e6:.1f}MB")
    
    return X_train, X_test, y_train, y_test 