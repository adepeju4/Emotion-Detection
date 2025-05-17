import uvicorn
from api import app

if __name__ == "__main__":
    print("Starting Emotion Detection API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 