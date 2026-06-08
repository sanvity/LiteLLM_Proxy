# ============================================================
# Stage 1 — Dependency builder
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build tools needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2 — Model cache pre-warmer
# Pre-downloads the DeBERTa PII model at build time so the
# container starts instantly without a 500MB cold-download.
# ============================================================
FROM builder AS model-warmer

WORKDIR /model_cache

# Set HuggingFace cache directory
ENV TRANSFORMERS_CACHE=/model_cache
ENV HF_HOME=/model_cache

# Pre-download the DeBERTa PII guardrail model
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForTokenClassification; \
model_name = 'Isotonic/deberta-v3-base_finetuned_ai4privacy_v2'; \
print('Downloading tokenizer...'); \
AutoTokenizer.from_pretrained(model_name, cache_dir='/model_cache'); \
print('Downloading model weights...'); \
AutoModelForTokenClassification.from_pretrained(model_name, cache_dir='/model_cache'); \
print('Model pre-cache complete.'); \
" || echo "Model pre-cache skipped (network unavailable at build time)"

# ============================================================
# Stage 3 — Runtime image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded model weights
COPY --from=model-warmer /model_cache /app/model_cache

# Copy application source code
COPY . .

# Environment configuration
# PORT is set by cloud platforms (Railway, Render, Fly.io) automatically
ENV PORT=8000
ENV HOST=0.0.0.0
ENV STREAMLIT_PORT=8501

# Tell HuggingFace to use the pre-cached model (no download on startup)
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_OFFLINE=1

# Streamlit config for headless reverse-proxy operation
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the single public port
EXPOSE $PORT

# Health check (used by Railway, Fly.io, Kubernetes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Single entrypoint — spawns Streamlit internally, serves everything on $PORT
CMD ["python", "main.py"]
