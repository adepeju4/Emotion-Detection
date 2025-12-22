import tensorflow as tf
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "fer2013": {
        "path": "models/fer2013_model.h5",
        "labels": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "input_shape": (48, 48)
    },
    "ckplus": {
        "path": "models/ckplus_model.h5",
        "labels": ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "input_shape": (48, 48)
    },
    "affectnet": {
        "path": "models/affectnet_model.h5",
        "labels": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
        "input_shape": (48, 48)
    }
}

models: Dict[str, Optional[tf.keras.Model]] = {}

def load_models():
    """Load all emotion detection models at startup."""
    global models
    for model_name, config in MODEL_CONFIGS.items():
        try:
            models[model_name] = tf.keras.models.load_model(config["path"])
            logger.info(f"Successfully loaded {model_name} model")
        except Exception as e:
            logger.error(f"Error loading {model_name} model: {e}")
            models[model_name] = None

def get_model(model_name: str) -> Optional[tf.keras.Model]:
    """Get a loaded model by name."""
    return models.get(model_name)

def get_model_config(model_name: str) -> Optional[dict]:
    """Get model configuration by name."""
    return MODEL_CONFIGS.get(model_name)

def get_all_models() -> Dict[str, Optional[tf.keras.Model]]:
    """Get all loaded models."""
    return models

def get_all_model_configs() -> Dict:
    """Get all model configurations."""
    return MODEL_CONFIGS

