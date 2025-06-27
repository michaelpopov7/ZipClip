# Use Python 3.11 with Ubuntu base for better compatibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    pkg-config \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (optional, for local LLM support)
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install wheel first
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies first
RUN pip install --no-cache-dir numpy>=1.19.0 imageio>=2.4.1 decorator>=4.4.2 proglog>=0.1.10 tqdm>=4.64.0 requests>=2.25.1

# Install imageio-ffmpeg separately
RUN pip install --no-cache-dir imageio-ffmpeg>=0.4.8

# Install moviepy separately
RUN pip install --no-cache-dir moviepy>=1.0.3

# Install remaining dependencies
RUN pip install --no-cache-dir openai-whisper>=20231117 openai>=1.0.0 PyYAML>=6.0 ollama>=0.1.0 flask>=2.3.0 werkzeug>=2.3.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads output static templates

# Set proper permissions
RUN chmod +x /app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "app.py"]