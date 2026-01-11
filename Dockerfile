FROM python:3.11-bookworm

WORKDIR /app

# Install ffmpeg runtime only (not dev headers)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using pre-built wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the multilingual model during build (faster cold starts)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

COPY main.py .

# Use python main.py which reads PORT from env
CMD ["python", "main.py"]
