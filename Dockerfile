FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Playwright's browser binaries are deliberately NOT installed here -
# ingestion/fetch_corpus.py (which needs them for JS-rendered sources) is a
# local/occasional tool, not part of the running API.
#
# If data/processed/chunks.jsonl is present in the build context, build the
# BM25 index and embeddings from it right here at image-build time - this is
# what makes a single self-contained image possible (e.g. for Hugging Face
# Spaces, which can't do the volume-mount trick docker-compose.yml uses for
# local dev). It's gitignored in this repo for local development (the local
# venv already has it), but a deployment target's own repo commits it
# specifically so this step has something to build from.
RUN if [ -f data/processed/chunks.jsonl ]; then \
      python ingestion/bm25_index.py && python ingestion/embed.py; \
    fi

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
