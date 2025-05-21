import os

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

def determine_num_classes(dataset_name):
    if dataset_name == 'fer2013':
        return 7
    elif dataset_name == 'ckplus':
        return 7
    elif dataset_name == 'affectnet':
        return 8
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        



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
        


