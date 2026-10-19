import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import (
    MODEL_VERSION,
    DECISION_THRESHOLD,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)
from app.predictor import get_predictor, AudioPredictor


# Modern FastAPI Lifespan: loads the model when server boots up
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model once into memory
    print("Initializing Audio Deepfake Predictor...")
    predictor = get_predictor()
    app.state.predictor = predictor
    print("Ready to receive requests.")
    yield
    # Shutdown logic (if any cleanup is needed in the future)
    print("Shutting down service...")


app = FastAPI(
    title="Audio Deepfake Detection API",
    description="Production-ready REST API for detecting synthetic/AI-generated speech using fine-tuned Wav2Vec2.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for Docker container and AWS health monitoring.
    Returns 200 only if the model is loaded and ready to serve requests.
    Returns 503 if the model failed to load at startup.
    """
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Service is not ready.",
        )
    return {
        "status": "healthy",
        "service": "audio-deepfake-detector",
        "model_version": MODEL_VERSION,
    }


@app.get("/version", status_code=status.HTTP_200_OK, tags=["Metadata"])
async def get_version_info():
    """
    Returns model metadata and decision threshold.
    """
    return {
        "model_version": MODEL_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
    }


@app.post("/predict", status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (.wav, .mp3, .flac, .m4a, .ogg) to detect if it is REAL or FAKE.
    """
    # 1. Validate file extension
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{file_ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # 2. Read file contents and validate file size
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb:.0f} MB.",
        )

    # 3. Run inference via the predictor
    predictor: AudioPredictor = getattr(app.state, "predictor", None) or get_predictor()
    try:
        result = predictor.predict(contents)
    except Exception as e:
        # Prevent server crash on corrupted/malformed audio data
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to decode or process audio data: {str(e)}",
        )

    # 4. Include filename in the response for tracking
    result["filename"] = file.filename
    return JSONResponse(content=result)
