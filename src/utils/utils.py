import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATASETS_DIR = os.path.join(BASE_DIR, 'data')


# Dataset paths
FER2013_DIR = os.path.join(DATASETS_DIR, 'Fer2013')
CKPLUS_DIR = os.path.join(DATASETS_DIR, 'CK+')
AFFECTNET_DIR = os.path.join(DATASETS_DIR, 'Affectnet')

# Model save paths
MODELS_SAVE_DIR = os.path.join(BASE_DIR, 'models')
FER2013_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'fer2013_model.h5')
CKPLUS_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'ckplus_model.h5')
AFFECTNET_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'affectnet_model.h5')
COMBINED_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'combined_model.h5') 

def get_model_dirs(dataset_name, model_type='cnn'):
    base_dir = os.path.join(MODELS_SAVE_DIR, model_type, dataset_name.lower())
    model_dir = os.path.join(base_dir, 'model')
    vis_dir = os.path.join(base_dir, 'visualizations')
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    return base_dir, model_dir, vis_dir

def setup_gpu_memory():
    """
    Configure GPU memory growth to avoid memory allocation issues.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPU devices found, using CPU")

def determine_num_classes(dataset_name):
    """
    Determine the number of emotion classes for each dataset.
    """
    if dataset_name.lower() == 'fer2013':
        return 7  # angry, disgust, fear, happy, sad, surprise, neutral
    elif dataset_name.lower() == 'ckplus':
        return 8  # angry, contempt, disgust, fear, happy, sad, surprise, neutral
    elif dataset_name.lower() == 'affectnet':
        return 8  # neutral, happy, sad, surprise, fear, disgust, anger, contempt
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_images_from_directory(directory, target_size=(48, 48)):
    """
    Load images from a directory and convert them to arrays.
    """
    images = []
    labels = []
    class_dirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(directory, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize(target_size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def load_dataset(dataset_name):
    """
    Load and preprocess the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to load ('fer2013', 'ckplus', or 'affectnet')
    
    Returns:
        X_train, y_train, X_test, y_test: Training and testing data
    """
    base_path = os.path.join('data', dataset_name)
    
    if not os.path.exists(base_path):
        raise ValueError(f"Dataset directory not found: {base_path}")
    
    if dataset_name.lower() == 'fer2013':
        # FER2013 has predefined train/test split
        train_path = os.path.join(base_path, 'train')
        test_path = os.path.join(base_path, 'test')
        
        X_train, y_train = load_images_from_directory(train_path)
        X_test, y_test = load_images_from_directory(test_path)
    
    else:  # For CK+ and AffectNet, create train/test split
        X, y = load_images_from_directory(base_path)
        
        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Split into train/test (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test

def create_results_directory(dataset_name):
    """
    Create directory structure for saving results.
    """
    base_dir = os.path.join('results', dataset_name)
    subdirs = ['training_history', 'confusion_matrices', 'feature_maps', 'metrics']
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    return base_dir

def get_emotion_mapping(dataset_name):
    if dataset_name == 'fer2013':
        return {
            'anger': 'anger',
            'disgust': 'disgust',
            'fear': 'fear',
            'happiness': 'happiness',
            'neutral': 'neutral',
            'sadness': 'sadness',
            'surprise': 'surprise'
        }
    elif dataset_name == 'ckplus':
        return {
            'anger': 'anger',
            'contempt': 'contempt',
            'disgust': 'disgust',
            'fear': 'fear',
            'happiness': 'happiness',
            'sadness': 'sadness',
            'surprise': 'surprise'
        }
    elif dataset_name == 'affectnet':
        return {
            'anger': 'anger',
            'contempt': 'contempt',
            'disgust': 'disgust',
            'fear': 'fear',
            'happiness': 'happiness',
            'neutral': 'neutral',
            'sadness': 'sadness',
            'surprise': 'surprise'
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        


