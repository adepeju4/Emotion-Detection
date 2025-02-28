import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

# Dataset paths
FER2013_DIR = os.path.join(DATASETS_DIR, 'Fer2013')
CKPLUS_DIR = os.path.join(DATASETS_DIR, 'CK+')
AFFECTNET_DIR = os.path.join(DATASETS_DIR, 'Affectnet')

# Image parameters
IMG_SIZE = 48  # Standard size for FER2013, we'll resize others to match
IMG_CHANNELS = 1  # Grayscale

# Training parameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# Emotion mapping to standardize labels across datasets
# Format: {dataset_name: {original_label: standardized_label}}
EMOTION_MAPPING = {
    'fer2013': {
        'angry': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happiness',
        'neutral': 'neutral',
        'sad': 'sadness',
        'surprise': 'surprise'
    },
    'ckplus': {
        'anger': 'anger',
        'contempt': 'contempt',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happiness',
        'sadness': 'sadness',
        'surprise': 'surprise'
    },
    'affectnet': {
        'anger': 'anger',
        'contempt': 'contempt',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happiness',
        'neutral': 'neutral',
        'sad': 'sadness',
        'surprise': 'surprise'
    }
}

# Standardized emotion classes for our models
# We'll use a subset of emotions that are common across all datasets
EMOTION_CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
# If you want to include all emotions:
# EMOTION_CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# Number of classes
NUM_CLASSES = len(EMOTION_CLASSES)

# Model save paths
MODELS_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
FER2013_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'fer2013_model.h5')
CKPLUS_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'ckplus_model.h5')
AFFECTNET_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'affectnet_model.h5')
COMBINED_MODEL_PATH = os.path.join(MODELS_SAVE_DIR, 'combined_model.h5') 