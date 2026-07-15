# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential needed for some compilation if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image for ultra-fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy requirements.txt to install dependencies
COPY requirements.txt .

# Install dependencies globally inside the container
RUN uv pip install --system --no-cache -r requirements.txt --index-strategy unsafe-best-match \
    && uv pip install --system --no-cache exrex

# Copy the rest of the application files
COPY . .

# Ensure correct directories exist for persistent storage
RUN mkdir -p /app/models /app/data

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Declare volumes for persistent models and generated evaluation data
VOLUME ["/app/models", "/app/data"]

# Default command starts the FastAPI server
CMD ["python", "main.py"]
