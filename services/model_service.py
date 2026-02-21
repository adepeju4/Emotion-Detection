import os
import logging
from typing import Dict, Optional, Any

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "fer2013": {
        "path": "models/fer2013_model.onnx",
        "labels": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "input_shape": (48, 48)
    },
    "ckplus": {
        "path": "models/ckplus_model.onnx",
        "labels": ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "input_shape": (48, 48)
    },
    "affectnet": {
        "path": "models/affectnet_model.onnx",
        "labels": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
        "input_shape": (48, 48)
    }
}

models: Dict[str, Optional[ort.InferenceSession]] = {}

def load_models():
    """Load all emotion detection models at startup."""
    logger.info("Models will be loaded lazily on first request")

def _load_model(model_name: str):
    """Load a single ONNX model from disk."""
    if model_name in models:
        return
    config = MODEL_CONFIGS.get(model_name)
    if config is None:
        return
    try:
        path = config["path"]
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            models[model_name] = None
            return
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        models[model_name] = session
        logger.info(f"Successfully loaded {model_name} model (ONNX)")
    except Exception as e:
        logger.error(f"Error loading {model_name} model: {e}")
        models[model_name] = None

def get_model(model_name: str) -> Optional[ort.InferenceSession]:
    """Get a model by name, loading it if needed."""
    if model_name not in models:
        _load_model(model_name)
    return models.get(model_name)

def predict(model_name: str, input_array: np.ndarray) -> np.ndarray:
    """Run inference on an input array using the named model."""
    session = get_model(model_name)
    if session is None:
        raise RuntimeError(f"Model '{model_name}' is not available")
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_array.astype(np.float32)})
    return result[0]

def get_model_config(model_name: str) -> Optional[dict]:
    """Get model configuration by name."""
    return MODEL_CONFIGS.get(model_name)

def get_all_models() -> Dict[str, Optional[Any]]:
    """Get all models, loading any that haven't been loaded yet."""
    for model_name in MODEL_CONFIGS:
        if model_name not in models:
            _load_model(model_name)
    return models

def get_all_model_configs() -> Dict:
    """Get all model configurations."""
    return MODEL_CONFIGS
