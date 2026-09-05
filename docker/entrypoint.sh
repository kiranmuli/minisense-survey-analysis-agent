#!/bin/sh
# ---------------------------------------------------------------------------
# Container startup:
#   1. wait for the Ollama service to be reachable
#   2. pull the chat + embedding models (idempotent; cached in a volume)
#   3. generate the survey dataset (once) and build the vector store (once)
#   4. launch the FastAPI app
# ---------------------------------------------------------------------------
set -e

BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434/v1}"
NATIVE="${BASE_URL%/v1}"                         # strip /v1 -> native Ollama API
MODEL="${OLLAMA_MODEL:-qwen3:8b}"
EMBED="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
RECORDS="${SURVEY_RECORDS:-60000}"

echo ">> Waiting for Ollama at ${NATIVE} ..."
until curl -sf "${NATIVE}/api/tags" >/dev/null 2>&1; do
  sleep 2
done
echo ">> Ollama is reachable."

echo ">> Pulling chat model: ${MODEL} (first run may take several minutes) ..."
curl -sf "${NATIVE}/api/pull" -d "{\"name\":\"${MODEL}\"}" >/dev/null \
  || echo "WARN: could not pull ${MODEL} (may already be present)"

echo ">> Pulling embedding model: ${EMBED} ..."
curl -sf "${NATIVE}/api/pull" -d "{\"name\":\"${EMBED}\"}" >/dev/null \
  || echo "WARN: could not pull ${EMBED} (may already be present)"

if [ ! -f data/survey_responses.json ]; then
  echo ">> Generating ${RECORDS} survey records ..."
  python data/generate_data.py --count "${RECORDS}"
else
  echo ">> Survey dataset already present — skipping generation."
fi

if [ ! -d chroma_db ]; then
  echo ">> Building FAQ vector store ..."
  python -m app.rag.ingest
else
  echo ">> Vector store already present — skipping ingest."
fi

echo ">> Starting API on :8000 ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
