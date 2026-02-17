FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Environment variables - comment shifted up to avoid syntax error with backslash
# Allow huggingface download during build (we bake the model anyway)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git build-essential \
    portaudio19-dev wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip & huggingface-cli
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel huggingface-hub[cli]

# Clone and install latest Fish-Speech (editable mode best)
RUN git clone https://github.com/fishaudio/fish-speech.git . && \
    pip3 install --no-cache-dir -e .

# Your custom requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake latest OpenAudio-S1-mini (recommended mini model Feb 2026)
RUN huggingface-cli download fishaudio/openaudio-s1-mini \
    --local-dir checkpoints/openaudio-s1-mini --include "*.pth" "*.json" && \
    ls -la checkpoints/openaudio-s1-mini  # Verify in build logs

COPY handler.py .
CMD ["python3", "-u", "handler.py"]