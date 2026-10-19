# 1. Base Image: Use official lightweight Python Linux image
FROM python:3.11-slim

# 2. Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# 3. Install system-level audio dependencies needed by soundfile & librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Set container working directory
WORKDIR /app

# 5. Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application source code and models
COPY app/ ./app/
COPY models/ ./models/

# 7. Expose API port
EXPOSE 8000

# 8. Health check for container orchestrators (AWS / Kubernetes / Docker)
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 9. Launch FastAPI server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
