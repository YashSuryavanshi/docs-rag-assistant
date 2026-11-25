"""Runtime configuration, read from environment variables (prefix RAG_) or a .env file.

Every tunable that affects retrieval quality lives here, so experiments
(eval/sweep_chunking.py) and deployments (k8s/configmap.yaml) change behaviour
without touching code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- corpus ---
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus" / "vllm"

    # --- chunking (measured, not guessed: see eval/sweep_chunking.py) ---
    chunk_size: int = 200  # words per chunk
    chunk_overlap: int = 40  # words carried over from the previous chunk

    # --- models ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models retrieve better when queries (not passages) carry this prefix.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- vector store ---
    # Set qdrant_url for a Qdrant server (Kubernetes); otherwise an embedded
    # on-disk store at qdrant_path is used (local development, CI).
    qdrant_url: str | None = None
    qdrant_path: Path = PROJECT_ROOT / "data" / "index" / "qdrant"
    collection: str = "vllm_docs"

    # --- embedding cache ---
    embedding_cache: Path = PROJECT_ROOT / "data" / "cache" / "embeddings.sqlite"

    # --- retrieval ---
    dense_k: int = 30  # candidates from vector search
    sparse_k: int = 30  # candidates from BM25
    rrf_k: int = 60  # Reciprocal Rank Fusion constant (60 is the value from the original paper)
    rerank_k: int = 30  # fused candidates passed to the cross-encoder
    final_k: int = 5  # passages returned / given to the LLM

    # --- generation (any OpenAI-compatible endpoint: vLLM, OpenAI, ...) ---
    llm_base_url: str | None = None  # e.g. http://localhost:8000/v1 for a vLLM server
    llm_model: str | None = None
    llm_api_key: str = "not-needed-for-vllm"
    llm_max_tokens: int = 512


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
