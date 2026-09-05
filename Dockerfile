# ============================================================================
# MiniSense — Survey Analysis Agent : application image
# Runs the FastAPI app. The LLM + embeddings are served by a separate Ollama
# container (see docker-compose.yml), which this app talks to over HTTP.
# ============================================================================
FROM python:3.11-slim

# curl is used by the entrypoint to pull models + health-check Ollama.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + the inputs needed to build data and the FAQ index.
COPY app/ ./app/
COPY data/generate_data.py ./data/generate_data.py
COPY data/product_faq.txt ./data/product_faq.txt
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh

EXPOSE 8000

# The entrypoint waits for Ollama, pulls models, generates data + vector store
# (once), then launches the API.
ENTRYPOINT ["./docker/entrypoint.sh"]
