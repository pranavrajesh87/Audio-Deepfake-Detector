import io
import soundfile as sf
import numpy as np


def test_health_endpoint(client):
    """Test that the /health endpoint responds with 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_version_endpoint(client):
    """Test that the /version endpoint returns configuration details."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "decision_threshold" in data
    assert ".wav" in data["supported_formats"]


def test_predict_unsupported_format(client):
    """Test that uploading a non-audio file returns HTTP 400."""
    fake_file = io.BytesIO(b"This is not audio content")
    response = client.post(
        "/predict",
        files={"file": ("malicious_script.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_empty_file(client):
    """Test that uploading an empty audio file returns HTTP 400."""
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/predict",
        files={"file": ("empty.wav", empty_file, "audio/wav")}
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_predict_valid_synthetic_audio(client):
    """
    Generate a 1-second synthetic 16 kHz sine wave in memory and verify /predict
    returns valid prediction fields without error.
    """
    sample_rate = 16000
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Generate 440 Hz tone
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Write wav bytes in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("synthetic_tone.wav", buffer, "audio/wav")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["real", "fake"]
    assert 0.0 <= data["score"] <= 1.0
    assert "threshold" in data
    assert "audio_duration_seconds" in data
    assert data["filename"] == "synthetic_tone.wav"
