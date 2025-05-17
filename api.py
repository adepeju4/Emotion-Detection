from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import config
from typing import Dict, List

app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions in facial images using multiple models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = {
    'fer2013': tf.keras.models.load_model(config.FER2013_MODEL_PATH),
    'ckplus': tf.keras.models.load_model(config.CKPLUS_MODEL_PATH),
    'affectnet': tf.keras.models.load_model(config.AFFECTNET_MODEL_PATH),
    'combined': tf.keras.models.load_model(config.COMBINED_MODEL_PATH)
}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess the uploaded image for model prediction."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to expected size
    resized = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE))
    
    # Normalize
    normalized = resized / 255.0
    
    # Reshape for model input
    preprocessed = normalized.reshape(1, config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
    
    return preprocessed

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Emotion Detection API",
        "available_models": list(models.keys()),
        "supported_emotions": {
            "fer2013": config.FER2013_EMOTIONS,
            "ckplus": config.CKPLUS_EMOTIONS,
            "affectnet": config.AFFECTNET_EMOTIONS,
            "combined": config.COMBINED_MODEL_EMOTIONS
        }
    }

@app.post("/detect/{model_name}")
async def detect_emotion(
    model_name: str,
    file: UploadFile = File(...),
) -> Dict[str, List[Dict[str, float]]]:
    """
    Detect emotions in the uploaded image using the specified model.
    
    Args:
        model_name: Name of the model to use (fer2013, ckplus, affectnet, or combined)
        file: Uploaded image file
    
    Returns:
        Dictionary containing emotion predictions and their confidence scores
    """
    if model_name not in models:
        return {"error": f"Model {model_name} not found. Available models: {list(models.keys())}"}
    
    # Read and preprocess the image
    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)
    
    # Get model predictions
    predictions = models[model_name].predict(preprocessed_image)[0]
    
    # Get emotion labels based on model
    if model_name == 'fer2013':
        emotions = config.FER2013_EMOTIONS
    elif model_name == 'ckplus':
        emotions = config.CKPLUS_EMOTIONS
    elif model_name == 'affectnet':
        emotions = config.AFFECTNET_EMOTIONS
    else:
        emotions = config.COMBINED_MODEL_EMOTIONS
    
    # Create results dictionary
    results = [
        {"emotion": emotion, "confidence": float(conf)}
        for emotion, conf in zip(emotions, predictions)
    ]
    
    # Sort by confidence
    results.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 