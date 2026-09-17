FROM python:3.9-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Set up a working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY api.py .

# Create a directory for outputs with wide permissions
RUN mkdir -p /tmp/stemsplitter_outputs && chmod 777 /tmp/stemsplitter_outputs

# Run the FastAPI server on Hugging Face's default port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
