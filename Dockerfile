# API image. Models are downloaded at build time so a starting pod does not
# depend on Hugging Face being reachable, and cold starts stay short.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding and reranker models into the image.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY rag/ rag/
COPY scripts/ scripts/
COPY data/corpus/MANIFEST.json data/corpus/MANIFEST.json

# data/cache must exist before the chown so a mounted volume inherits app ownership.
RUN mkdir -p /app/data/cache && useradd --create-home --uid 10001 app && chown -R app /app /opt/hf
USER app

EXPOSE 8080
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8080"]
