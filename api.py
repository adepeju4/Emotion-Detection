from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

from services.model_service import load_models
from services.face_detection_service import load_face_detector
from controllers.emotion_controller import predict_emotions, get_models_info, get_health_status
from controllers.dictionary_controller import get_word_definition

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_models()
load_face_detector()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("fer2013")
):
    """Detect faces in an image and predict emotions for each face using the specified model."""
    return await predict_emotions(file, model)

@app.get("/models")
async def get_models():
    """Get information about available models."""
    return get_models_info()

@app.get("/health")
async def health_check():
    """Check API health and model status."""
    return get_health_status()

@app.get("/dictionary-definition/{word}")
async def get_dictionary_definition(word: str):
    """Proxy endpoint to fetch dictionary definition from Free Dictionary API."""
    return await get_word_definition(word)

@app.get("/")
async def root():
    return {
        "message": "Emotion Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/predict - POST - Detect emotions in an image",
            "/models - GET - Get available models info",
            "/health - GET - Check API health",
            "/dictionary-definition/{word} - GET - Get dictionary definition"
        ]
    }
