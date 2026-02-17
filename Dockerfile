FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=0  # Allow download if needed, but we bake it

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git build-essential \
    portaudio19-dev wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip & tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel huggingface-hub[cli]

# Clone and install latest Fish-Speech
RUN git clone https://github.com/fishaudio/fish-speech.git . && \
    pip3 install --no-cache-dir -e .

# Install your custom requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the latest recommended mini model (OpenAudio-S1-mini)
RUN huggingface-cli download fishaudio/openaudio-s1-mini \
    --local-dir checkpoints/s1-mini --include "*.pth" "*.json" --exclude "*large*" && \
    ls -la checkpoints/s1-mini  # Just to verify in build logs

COPY handler.py .
CMD ["python3", "-u", "handler.py"]