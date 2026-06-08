# ============================================================
# LiteLLM Gateway — FastAPI Service
# Single-stage build for the API backend.
# The Streamlit UI runs as a separate service.
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Environment defaults ($PORT is injected by Railway/Fly/Render)
ENV PORT=8000
ENV HOST=0.0.0.0

# HuggingFace model cache (DeBERTa PII model)
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Pre-download DeBERTa-v3 PII model weights into image layer
RUN python -c "from transformers import pipeline; pipeline('token-classification', model='Isotonic/deberta-v3-base_finetuned_ai4privacy_v2')"


# Expose port (Railway overrides this with $PORT)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python", "main.py"]
