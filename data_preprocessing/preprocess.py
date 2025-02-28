import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import config

def load_fer2013():
    """Load FER2013 dataset which is already split into train/test"""
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # Process training data
    train_dir = os.path.join(config.FER2013_DIR, 'train')
    for emotion in os.listdir(train_dir):
        if emotion.startswith('.'):  # Skip hidden files
            continue
            
        emotion_dir = os.path.join(train_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
            
        # Map the emotion to standardized label
        std_emotion = config.EMOTION_MAPPING['fer2013'].get(emotion)
        if std_emotion not in config.EMOTION_CLASSES:
            continue  # Skip emotions not in our target classes
            
        emotion_idx = config.EMOTION_CLASSES.index(std_emotion)
        
        for img_file in os.listdir(emotion_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # Resize if needed
            if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                
            # Normalize pixel values
            img = img / 255.0
            
            train_data.append(img)
            train_labels.append(emotion_idx)
    
    # Process test data
    test_dir = os.path.join(config.FER2013_DIR, 'test')
    for emotion in os.listdir(test_dir):
        if emotion.startswith('.'):  # Skip hidden files
            continue
            
        emotion_dir = os.path.join(test_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
            
        # Map the emotion to standardized label
        std_emotion = config.EMOTION_MAPPING['fer2013'].get(emotion)
        if std_emotion not in config.EMOTION_CLASSES:
            continue  # Skip emotions not in our target classes
            
        emotion_idx = config.EMOTION_CLASSES.index(std_emotion)
        
        for img_file in os.listdir(emotion_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # Resize if needed
            if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                
            # Normalize pixel values
            img = img / 255.0
            
            test_data.append(img)
            test_labels.append(emotion_idx)
    
    # Convert to numpy arrays
    X_train = np.array(train_data).reshape(-1, config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
    y_train = to_categorical(np.array(train_labels), num_classes=config.NUM_CLASSES)
    
    X_test = np.array(test_data).reshape(-1, config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
    y_test = to_categorical(np.array(test_labels), num_classes=config.NUM_CLASSES)
    
    return (X_train, y_train), (X_test, y_test)

def load_ckplus():
    """Load CK+ dataset and split into train/validation"""
    data = []
    labels = []
    
    for emotion in os.listdir(config.CKPLUS_DIR):
        if emotion.startswith('.'):  # Skip hidden files
            continue
            
        emotion_dir = os.path.join(config.CKPLUS_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            continue
            
        # Map the emotion to standardized label
        std_emotion = config.EMOTION_MAPPING['ckplus'].get(emotion)
        if std_emotion not in config.EMOTION_CLASSES:
            continue  # Skip emotions not in our target classes
            
        emotion_idx = config.EMOTION_CLASSES.index(std_emotion)
        
        for img_file in os.listdir(emotion_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Convert to grayscale if needed
            if config.IMG_CHANNELS == 1 and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            # Resize if needed
            if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                
            # Normalize pixel values
            img = img / 255.0
            
            data.append(img)
            labels.append(emotion_idx)
    
    # Convert to numpy arrays
    X = np.array(data).reshape(-1, config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
    y = to_categorical(np.array(labels), num_classes=config.NUM_CLASSES)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    return (X_train, y_train), (X_val, y_val)

def load_affectnet():
    """Load AffectNet dataset and split into train/validation"""
    data = []
    labels = []
    
    # Check if labels.csv exists and use it if available
    labels_csv_path = os.path.join(config.AFFECTNET_DIR, 'labels.csv')
    if os.path.exists(labels_csv_path):
        # Load from CSV
        df = pd.read_csv(labels_csv_path)
        # Process according to CSV structure
        # This is a placeholder - adjust based on your actual CSV structure
        for _, row in df.iterrows():
            img_path = os.path.join(config.AFFECTNET_DIR, row['path'])
            emotion = row['emotion']
            
            # Map the emotion to standardized label
            std_emotion = config.EMOTION_MAPPING['affectnet'].get(emotion)
            if std_emotion not in config.EMOTION_CLASSES:
                continue  # Skip emotions not in our target classes
                
            emotion_idx = config.EMOTION_CLASSES.index(std_emotion)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Convert to grayscale if needed
            if config.IMG_CHANNELS == 1 and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            # Resize if needed
            if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                
            # Normalize pixel values
            img = img / 255.0
            
            data.append(img)
            labels.append(emotion_idx)
    else:
        # Load directly from directory structure
        for emotion in os.listdir(config.AFFECTNET_DIR):
            if emotion.startswith('.') or emotion == 'labels.csv':  # Skip hidden files and CSV
                continue
                
            emotion_dir = os.path.join(config.AFFECTNET_DIR, emotion)
            if not os.path.isdir(emotion_dir):
                continue
                
            # Map the emotion to standardized label
            std_emotion = config.EMOTION_MAPPING['affectnet'].get(emotion)
            if std_emotion not in config.EMOTION_CLASSES:
                continue  # Skip emotions not in our target classes
                
            emotion_idx = config.EMOTION_CLASSES.index(std_emotion)
            
            for img_file in os.listdir(emotion_dir):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(emotion_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                    
                # Convert to grayscale if needed
                if config.IMG_CHANNELS == 1 and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                # Resize if needed
                if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
                    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                    
                # Normalize pixel values
                img = img / 255.0
                
                data.append(img)
                labels.append(emotion_idx)
    
    # Convert to numpy arrays
    X = np.array(data).reshape(-1, config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
    y = to_categorical(np.array(labels), num_classes=config.NUM_CLASSES)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    return (X_train, y_train), (X_val, y_val)

def load_combined_dataset():
    """Load and combine all three datasets"""
    # Load individual datasets
    (fer_train, fer_train_labels), (fer_test, fer_test_labels) = load_fer2013()
    (ck_train, ck_train_labels), (ck_val, ck_val_labels) = load_ckplus()
    (affectnet_train, affectnet_train_labels), (affectnet_val, affectnet_val_labels) = load_affectnet()
    
    # Combine training data
    X_train = np.concatenate([fer_train, ck_train, affectnet_train], axis=0)
    y_train = np.concatenate([fer_train_labels, ck_train_labels, affectnet_train_labels], axis=0)
    
    # Combine validation data
    X_val = np.concatenate([fer_test, ck_val, affectnet_val], axis=0)
    y_val = np.concatenate([fer_test_labels, ck_val_labels, affectnet_val_labels], axis=0)
    
    return (X_train, y_train), (X_val, y_val)

def apply_data_augmentation(X_train, y_train, augmentation_factor=2):
    """Apply data augmentation to increase training data and balance classes"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Create an image data generator with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Count samples per class
    class_counts = np.sum(y_train, axis=0)
    max_count = np.max(class_counts)
    
    X_augmented = []
    y_augmented = []
    
    # Add original data
    X_augmented.append(X_train)
    y_augmented.append(y_train)
    
    # Generate augmented data for underrepresented classes
    for class_idx in range(config.NUM_CLASSES):
        class_indices = np.where(y_train[:, class_idx] == 1)[0]
        n_samples = len(class_indices)
        
        if n_samples == 0:
            continue
            
        # Calculate how many augmented samples we need
        n_to_generate = min(int(max_count * augmentation_factor) - n_samples, n_samples * (augmentation_factor - 1))
        
        if n_to_generate <= 0:
            continue
            
        # Select samples for this class
        X_class = X_train[class_indices]
        y_class = y_train[class_indices]
        
        # Generate augmented samples
        X_aug = []
        batch_size = min(n_samples, 32)  # Process in batches to avoid memory issues
        
        # Configure the data generator
        datagen.fit(X_class)
        
        # Generate augmented samples
        aug_count = 0
        for X_batch, y_batch in datagen.flow(X_class, y_class, batch_size=batch_size):
            X_aug.append(X_batch)
            aug_count += len(X_batch)
            if aug_count >= n_to_generate:
                break
                
        # Concatenate augmented samples
        X_aug = np.concatenate(X_aug, axis=0)[:n_to_generate]
        y_aug = np.tile(np.eye(config.NUM_CLASSES)[class_idx], (len(X_aug), 1))
        
        X_augmented.append(X_aug)
        y_augmented.append(y_aug)
    
    # Combine all data
    X_augmented = np.concatenate(X_augmented, axis=0)
    y_augmented = np.concatenate(y_augmented, axis=0)
    
    # Shuffle the data
    indices = np.arange(len(X_augmented))
    np.random.shuffle(indices)
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]
    
    return X_augmented, y_augmented 