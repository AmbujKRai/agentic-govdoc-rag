FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Playwright's browser binaries are deliberately NOT installed here -
# ingestion/fetch_corpus.py (which needs them for JS-rendered sources) is a
# local/occasional tool, not part of the running API. Pre-built indexes
# (data/qdrant_db or a QDRANT_URL server, data/bm25_index.pkl,
# data/processed/chunks.jsonl) are provided via volume mount in
# docker-compose.yml rather than baked in, keeping this image lean.

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
