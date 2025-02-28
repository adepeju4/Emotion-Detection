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
    
    # Use CK+ specific emotions instead of general EMOTION_CLASSES
    valid_emotions = config.CKPLUS_EMOTIONS
    print(f"Loading CK+ dataset with emotions: {valid_emotions}")
    
    for emotion in os.listdir(config.CKPLUS_DIR):
        if emotion.startswith('.'):  # Skip hidden files
            continue
            
        emotion_dir = os.path.join(config.CKPLUS_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            continue
            
        # Map the emotion to standardized label
        std_emotion = config.EMOTION_MAPPING['ckplus'].get(emotion)
        
        # Skip emotions not in our CK+ emotions list
        if std_emotion not in valid_emotions:
            print(f"Skipping emotion {emotion} (maps to {std_emotion}) - not in valid emotions list")
            continue
            
        emotion_idx = valid_emotions.index(std_emotion)
        print(f"Processing {emotion} (maps to {std_emotion}, index {emotion_idx})")
        
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
    
    # One-hot encode the labels using the CK+ emotions length
    y = np.zeros((len(labels), len(valid_emotions)))
    for i, label in enumerate(labels):
        y[i, label] = 1
    
    print(f"Loaded {len(X)} images from CK+ dataset")
    print(f"Emotion distribution: {np.sum(y, axis=0)}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=42, stratify=labels
    )
    
    return (X_train, y_train), (X_val, y_val)

def load_affectnet():
    """Load and preprocess the AffectNet dataset"""
    print("Loading AffectNet dataset...")
    
    # Check if the directory exists
    if not os.path.exists(config.AFFECTNET_DIR):
        print(f"Error: AffectNet directory not found at {config.AFFECTNET_DIR}")
        print("Please download the dataset and extract it to this location.")
        # Return empty arrays to avoid crashing the program
        return (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    # Check for CSV file with annotations
    csv_path = os.path.join(config.AFFECTNET_DIR, 'annotations.csv')
    if not os.path.exists(csv_path):
        # Try alternative filenames
        alternative_paths = [
            os.path.join(config.AFFECTNET_DIR, 'affectnet_annotations.csv'),
            os.path.join(config.AFFECTNET_DIR, 'metadata.csv'),
            os.path.join(config.AFFECTNET_DIR, 'labels.csv')
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                csv_path = alt_path
                break
        else:
            print(f"Error: AffectNet annotations file not found.")
            print("Please make sure the dataset includes a CSV file with image paths and emotion labels.")
            # Return empty arrays to avoid crashing the program
            return (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    # Load annotations
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded AffectNet annotations with {len(df)} entries")
        
        # Print column names for debugging
        print(f"Available columns in CSV: {df.columns.tolist()}")
        
        # Determine the image path column name
        path_column = None
        possible_path_columns = ['path', 'image', 'file', 'filename', 'subDirectory_filePath']
        for col in possible_path_columns:
            if col in df.columns:
                path_column = col
                print(f"Using '{path_column}' as the image path column")
                break
        
        if path_column is None:
            print("Error: Could not find image path column in CSV file")
            print(f"Available columns: {df.columns.tolist()}")
            # Return empty arrays to avoid crashing the program
            return (np.array([]), np.array([])), (np.array([]), np.array([]))
        
        # Determine the emotion column name
        emotion_column = None
        possible_emotion_columns = ['emotion', 'expression', 'label', 'class', 'expression_label']
        for col in possible_emotion_columns:
            if col in df.columns:
                emotion_column = col
                print(f"Using '{emotion_column}' as the emotion column")
                break
        
        if emotion_column is None:
            print("Error: Could not find emotion column in CSV file")
            print(f"Available columns: {df.columns.tolist()}")
            # Return empty arrays to avoid crashing the program
            return (np.array([]), np.array([])), (np.array([]), np.array([]))
        
        # Process images and labels
        images = []
        labels = []
        
        # Map AffectNet emotion codes to our standardized labels
        emotion_map = config.EMOTION_MAPPING['affectnet']
        
        # Get list of valid emotions for this dataset
        valid_emotions = config.AFFECTNET_EMOTIONS
        
        # Process each row in the CSV
        for idx, row in df.iterrows():
            try:
                # Get image path
                img_path = os.path.join(config.AFFECTNET_DIR, row[path_column])
                
                # Check if file exists
                if not os.path.exists(img_path):
                    # Try looking in subdirectories
                    if os.path.exists(os.path.join(config.AFFECTNET_DIR, 'images', row[path_column])):
                        img_path = os.path.join(config.AFFECTNET_DIR, 'images', row[path_column])
                    else:
                        print(f"Warning: Image file not found: {img_path}")
                        continue
                
                # Get emotion label
                emotion = row[emotion_column]
                
                # Convert to standardized emotion label if needed
                if isinstance(emotion, str) and emotion in emotion_map:
                    emotion = emotion_map[emotion]
                elif isinstance(emotion, (int, float)):
                    # If emotion is a number, map it to the corresponding emotion
                    emotion_idx = int(emotion)
                    if emotion_idx < len(valid_emotions):
                        emotion = valid_emotions[emotion_idx]
                    else:
                        print(f"Warning: Unknown emotion index: {emotion_idx}")
                        continue
                
                # Skip if emotion is not in our list
                if emotion not in valid_emotions:
                    continue
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image: {img_path}")
                    continue
                
                # Convert to grayscale if needed
                if config.IMG_CHANNELS == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize image
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                
                # Normalize pixel values
                img = img / 255.0
                
                # Add channel dimension if grayscale
                if config.IMG_CHANNELS == 1:
                    img = np.expand_dims(img, axis=-1)
                
                # Add to lists
                images.append(img)
                
                # One-hot encode the label
                label = np.zeros(len(valid_emotions))
                label[valid_emotions.index(emotion)] = 1
                labels.append(label)
                
                # Print progress
                if (idx + 1) % 1000 == 0:
                    print(f"Processed {idx + 1} images")
            
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Convert lists to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images from AffectNet dataset")
        
        # Split into train and validation sets
        split_idx = int(len(X) * (1 - config.VALIDATION_SPLIT))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return (X_train, y_train), (X_val, y_val)
    
    except Exception as e:
        print(f"Error loading AffectNet dataset: {e}")
        import traceback
        traceback.print_exc()
        # Return empty arrays to avoid crashing the program
        return (np.array([]), np.array([])), (np.array([]), np.array([]))

def load_combined_dataset():
    """Load and combine all three datasets"""
    print("Loading combined dataset...")
    
    # Save original emotion classes
    original_emotion_classes = config.EMOTION_CLASSES
    
    # Temporarily set emotion classes for each dataset
    config.EMOTION_CLASSES = config.FER2013_EMOTIONS
    (fer_train, fer_train_labels), (fer_test, fer_test_labels) = load_fer2013()
    
    config.EMOTION_CLASSES = config.CKPLUS_EMOTIONS
    (ck_train, ck_train_labels), (ck_val, ck_val_labels) = load_ckplus()
    
    config.EMOTION_CLASSES = config.AFFECTNET_EMOTIONS
    (affectnet_train, affectnet_train_labels), (affectnet_val, affectnet_val_labels) = load_affectnet()
    
    # Restore original emotion classes
    config.EMOTION_CLASSES = original_emotion_classes
    
    # Get the combined set of emotions
    combined_emotions = config.COMBINED_MODEL_EMOTIONS
    print(f"Combined emotions: {combined_emotions}")
    
    # Function to remap labels to combined emotion set
    def remap_labels(labels, source_emotions, target_emotions):
        new_labels = np.zeros((labels.shape[0], len(target_emotions)))
        for i in range(labels.shape[0]):
            # Find which emotion is active in this sample
            active_idx = np.argmax(labels[i])
            if active_idx < len(source_emotions):
                emotion = source_emotions[active_idx]
                if emotion in target_emotions:
                    new_idx = target_emotions.index(emotion)
                    new_labels[i, new_idx] = 1
        return new_labels
    
    # Remap labels to combined emotion set
    fer_train_labels_remapped = remap_labels(fer_train_labels, config.FER2013_EMOTIONS, combined_emotions)
    fer_test_labels_remapped = remap_labels(fer_test_labels, config.FER2013_EMOTIONS, combined_emotions)
    
    ck_train_labels_remapped = remap_labels(ck_train_labels, config.CKPLUS_EMOTIONS, combined_emotions)
    ck_val_labels_remapped = remap_labels(ck_val_labels, config.CKPLUS_EMOTIONS, combined_emotions)
    
    affectnet_train_labels_remapped = remap_labels(affectnet_train_labels, config.AFFECTNET_EMOTIONS, combined_emotions)
    affectnet_val_labels_remapped = remap_labels(affectnet_val_labels, config.AFFECTNET_EMOTIONS, combined_emotions)
    
    # Combine training data
    X_train = np.concatenate([fer_train, ck_train, affectnet_train], axis=0)
    y_train = np.concatenate([fer_train_labels_remapped, ck_train_labels_remapped, affectnet_train_labels_remapped], axis=0)
    
    # Combine validation data
    X_val = np.concatenate([fer_test, ck_val, affectnet_val], axis=0)
    y_val = np.concatenate([fer_test_labels_remapped, ck_val_labels_remapped, affectnet_val_labels_remapped], axis=0)
    
    print(f"Combined training data shape: {X_train.shape}")
    print(f"Combined validation data shape: {X_val.shape}")
    print(f"Emotion distribution in training data: {np.sum(y_train, axis=0)}")
    
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