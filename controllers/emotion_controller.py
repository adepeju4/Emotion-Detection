from fastapi import File, UploadFile, Form, HTTPException
import time
import logging

from services.emotion_service import validate_image, detect_and_analyze_faces
from services.model_service import MODEL_CONFIGS, get_all_models

logger = logging.getLogger(__name__)

async def predict_emotions(file: UploadFile, model: str):
    """Predict emotions in an uploaded image."""
    if model not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model name. Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    
    models = get_all_models()
    if models[model] is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{model}' is not available"
        )
    
    try:
        contents = await file.read()
        if len(contents) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        start_time = time.time()
        image = validate_image(contents)
        face_results = detect_and_analyze_faces(image, model)
        processing_time = time.time() - start_time
        
        logger.info(f"Processed image with {len(face_results)} faces in {processing_time:.3f} seconds")
        
        return {
            "model": model,
            "processing_time": processing_time,
            "faces_detected": len(face_results),
            "faces": face_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def get_models_info():
    """Get information about available models."""
    models = get_all_models()
    return {
        name: {
            "available": models[name] is not None,
            "labels": config["labels"],
            "input_shape": config["input_shape"]
        }
        for name, config in MODEL_CONFIGS.items()
    }

def get_health_status():
    """Check API health and model status."""
    models = get_all_models()
    return {
        "status": "healthy",
        "models": {
            name: "loaded" if model is not None else "not_loaded"
            for name, model in models.items()
        }
    }
