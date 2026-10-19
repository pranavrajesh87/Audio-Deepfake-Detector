# Audio Deepfake Detection Microservice

A production-ready REST API for detecting synthetic and AI-generated speech using a fine-tuned **Wav2Vec2** architecture, accompanied by cross-era stress testing that exposes real-world domain shift limitations.

---

## Architecture Overview

```
                      +-----------------------------------+
                      |   Client Audio Upload (Multipart) |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------+-----------------+
                      |     FastAPI Gateway (/predict)    |
                      |  - Format Validation (.wav/.mp3)  |
                      |  - 10MB Size Ceiling Check        |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------+-----------------+
                      |    Audio Preprocessing (Librosa)  |
                      |  - Mono conversion @ 16 kHz       |
                      |  - Standardized to 64,000 samples |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------+-----------------+
                      |    Wav2Vec2 Feature Extractor     |
                      |  - Time-distributed frames (768d) |
                      |  - Temporal Mean-Pooling          |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------+-----------------+
                      |      Classification Head          |
                      |  - Linear(768->256) -> ReLU       |
                      |  - Dropout(0.1) -> Linear(256->1) |
                      |  - Sigmoid Prob >= 0.90 -> FAKE   |
                      +-----------------------------------+
```

---

## Key Research Findings: The "Studio" Domain Shift

| Test Group | Result | Interpretation |
|---|---|---|
| **ASVspoof 2019 Eval (Baseline)** | EER: **4.26%**, Acc: **84.2%** | Strong in-distribution performance |
| **ASVspoof Real Voices (Control)** | **9/10** detected as REAL (99.6% conf) | Model accurately detects clean, studio-matched human speech |
| **Edge TTS Neural (Modern Fakes)** | **100%** detected as FAKE | Effectively identifies modern neural text-to-speech synthesis |
| **Self-Recorded Laptop Mic Voice** | **0/10** detected as REAL (99.7% "fake") | **Domain shift:** Model learned studio acoustic artifacts rather than pure vocal biometrics |

> **Key Takeaway**: Fine-tuned acoustic models can overfit to background studio conditions. Stress-testing against out-of-domain acoustic environments is crucial for production deployment.

---

## API Endpoints

### 1. Health Probe
```http
GET /health
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "audio-deepfake-detector",
  "model_version": "v1"
}
```

### 2. Service Metadata
```http
GET /version
```
**Response (200 OK):**
```json
{
  "model_version": "v1",
  "decision_threshold": 0.9,
  "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
  "max_file_size_mb": 10.0
}
```

### 3. Predict Deepfake
```http
POST /predict
Content-Type: multipart/form-data
```
**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@sample_audio.wav"
```

**Response (200 OK):**
```json
{
  "prediction": "fake",
  "score": 0.9842,
  "threshold": 0.9,
  "model_version": "v1",
  "audio_duration_seconds": 3.42,
  "filename": "sample_audio.wav"
}
```

---

## Local Development & Setup

### 1. Prerequisites
* **Python 3.10+**
* **ffmpeg** (required for decoding compressed audio formats such as `.mp3`, `.m4a`):
  * **Windows**: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
  * **macOS**: `brew install ffmpeg`
  * **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1`

### 2. Clone & Setup Virtual Environment
```bash
git clone https://github.com/pranavrajesh87/Audio-Deepfake-Detector.git
cd Audio-Deepfake-Detector

# Create virtual environment
python -m venv venv

# Activate virtual environment:
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (Command Prompt):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Model Weights (Zero Setup Auto-Download)
The 378 MB fine-tuned Wav2Vec2 checkpoint is hosted on [GitHub Releases (v1.0.0)](https://github.com/pranavrajesh87/Audio-Deepfake-Detector/releases/tag/v1.0.0).

* **Automatic**: When you start the API server, the microservice will automatically detect if `models/best_model.pt` is missing and stream it down with a progress bar.
* **Manual (Optional)**: If you prefer manual download, save `best_model.pt` directly into `models/`:
  ```bash
  # Linux/macOS curl:
  curl -L -o models/best_model.pt https://github.com/pranavrajesh87/Audio-Deepfake-Detector/releases/download/v1.0.0/best_model.pt
  ```

### 4. Run Automated Tests
```bash
python -m pytest
```

### 5. Start the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI will be available at: **`http://localhost:8000/docs`**

---

## Running with Docker

### 1. Build the Docker Image
```bash
docker build -t audio-deepfake-detector:latest .
```

### 2. Run the Container
You can run the container with a mounted `models` volume to persist the downloaded weights on your host machine:

```bash
# Windows PowerShell / macOS / Linux:
docker run -d -p 8000:8000 -v ${PWD}/models:/app/models --name deepfake-detector audio-deepfake-detector:latest
```

*(Note: On first startup, if `models/best_model.pt` is not present, the container will automatically download it from GitHub Releases into the mounted directory and initialize the model).*

### 3. Verify Health & Logs
```bash
docker logs -f deepfake-detector
curl http://localhost:8000/health
```

---

## CI/CD Pipeline
Continuous integration and continuous deployment are configured via GitHub Actions (`.github/workflows/ci.yml`):
* Automated Pytest suite executes across all pull requests and main branch commits.
* Passing builds on `main` automatically publish production container images to GitHub Container Registry (`ghcr.io`).