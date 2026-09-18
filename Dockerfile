FROM python:3.9-slim

# Install system dependencies and build tools for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create output directory for static file serving with wide permissions
RUN mkdir -p /tmp/stemsplitter_outputs && chmod -R 777 /tmp/stemsplitter_outputs

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
