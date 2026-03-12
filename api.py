from fastapi import FastAPI, File, UploadFile, Form, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from dotenv import load_dotenv

from services.model_service import load_models
from services.face_detection_service import load_face_detector
from controllers.emotion_controller import predict_emotions, get_models_info, get_health_status
from controllers.dictionary_controller import get_word_definition

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, headers_enabled=True)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("30/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("fer2013")
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
@limiter.limit("60/minute")
async def get_dictionary_definition(request: Request, word: str):
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
