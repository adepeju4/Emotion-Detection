from fastapi import FastAPI, File, UploadFile, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
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

router = APIRouter(prefix="/api")

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("fer2013"),
    response: Response = None,
):
    """Detect faces in an image and predict emotions for each face using the specified model."""
    return await predict_emotions(file, model)

@router.get("/models")
async def get_models():
    """Get information about available models."""
    return get_models_info()

@router.get("/health")
async def health_check():
    """Check API health and model status."""
    return get_health_status()

@router.get("/dictionary-definition/{word}")
async def get_dictionary_definition(word: str):
    """Proxy endpoint to fetch dictionary definition from Free Dictionary API."""
    return await get_word_definition(word)

app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "Emotion Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/api/predict - POST - Detect emotions in an image",
            "/api/models - GET - Get available models info",
            "/api/health - GET - Check API health",
            "/api/dictionary-definition/{word} - GET - Get dictionary definition"
        ]
    }
