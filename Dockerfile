# Use a slim but recent Python image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_FIND_LINKS=""

WORKDIR /app

# Install system deps required for scientific libs (torch, sentence-transformers, chromadb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --timeout=300 --retries=5 --default-timeout=300 -r requirements.txt

# Copy the application
COPY . .

# Cloud Run injects $PORT automatically
ENV PORT=8080
EXPOSE 8080

# Use Uvicorn with recommended production flags
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]