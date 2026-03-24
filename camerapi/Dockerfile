# Dockerfile for Server 1 (Camera Service - Raspberry Pi 3)
# Optimized for ARM architecture and low memory footprint

FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for temporary storage
RUN mkdir -p /tmp/camera_captures

# Environment variables (can be overridden in docker-compose)
ENV PYTHONUNBUFFERED=1
ENV USE_CAMERA=true
ENV CAMERA_INDEX=0
ENV SERVER2_URL=http://server2:8002
ENV TEMP_DIR=/tmp/camera_captures
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8001

# ❌ Healthcheck hier raus, weil das Image auch für den Button-Listener benutzt wird
# HEALTHCHECK ...  --> wird jetzt in docker-compose pro Service gesetzt

# Default: Camera-API starten
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
