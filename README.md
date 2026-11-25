# docs-rag-assistant

Question answering over the [vLLM documentation](https://github.com/vllm-project/vllm/tree/main/docs):
hybrid retrieval, cross-encoder reranking, incremental indexing, and a retrieval-quality gate that runs in CI.

```
question
   ├─► dense search (bge-small, Qdrant)  top 30 ─┐
   └─► BM25 keyword search               top 30 ─┴─► Reciprocal Rank Fusion ─► cross-encoder rerank ─► top 5 ─► LLM (cited answer)
```

## Measured results

36 hand-labelled questions, top-5 passages, on the corpus pinned in `data/corpus/MANIFEST.json`
(243 files, 2,628 chunks). A passage counts as relevant if it comes from a file that answers the question.

| retrieval mode | hit@5 | MRR | context precision | context recall | p50 latency (CPU) |
|---|---|---|---|---|---|
| dense only | 0.972 | 0.859 | 0.812 | 0.833 | 29 ms |
| BM25 only | 0.917 | 0.654 | 0.637 | 0.806 | 9 ms |
| hybrid (dense + BM25, RRF) | 0.944 | 0.831 | 0.763 | 0.861 | 39 ms |
| **hybrid + rerank** | **1.000** | **0.905** | **0.862** | **0.917** | 3,583 ms |

What this shows:

- **Hybrid without reranking is not better than dense alone.** Fusing in BM25 raises recall (0.833 to 0.861)
  because keyword search surfaces files that embeddings miss, but it also lets noisy keyword hits into the top
  ranks, so MRR and precision drop. The reranker is what turns the extra candidates into a gain.
- **Reranking is the expensive stage**: roughly 100x the latency of everything else on CPU, because the
  cross-encoder scores 30 query-passage pairs per question. A GPU, or a smaller `RAG_RERANK_K`, cuts it.
- **Keyword search earns its place on identifiers.** Queries such as `--tool-call-parser` or
  `VLLM_LOGGING_LEVEL=DEBUG` are tokenised so flags stay whole, which embeddings alone handle poorly.

### Limits of this evaluation (read before quoting the numbers)

- 36 questions is small. `hit@5 = 1.000` against `0.972` for dense is a one-question difference; the MRR and
  precision gaps are the more reliable signal.
- The questions and relevance labels were written by the author and checked against the corpus with text search,
  not by an independent annotator. Labels are at file level, not passage level.
- Answer faithfulness (`eval/faithfulness.py`) is implemented but needs an LLM endpoint and has **not been run yet**.
- Latency is from a laptop CPU and says nothing about GPU or server hardware.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

python scripts/fetch_corpus.py --ref pinned                 # the exact docs commit the eval labels refer to
python -m rag index                                         # first run embeds everything (about 6 min on CPU)
python -m rag ask "How do I enable automatic prefix caching?"
uvicorn rag.api:app --port 8080                             # POST /ask, GET /health/live, /health/ready
```

To get generated answers, point it at any OpenAI-compatible endpoint, for example your own vLLM server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
export RAG_LLM_BASE_URL=http://localhost:8000/v1 RAG_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Without one, the API still returns the retrieved passages and `answer: null`.

## Evaluation

```bash
python -m eval.run_eval                              # compare all four retrieval modes
python -m eval.run_eval --mode hybrid_rerank --gate  # exit 1 if below eval/thresholds.yaml (the CI gate)
python -m eval.run_eval --faithfulness               # judge generated answers (needs an LLM)
python -m eval.sweep_chunking                        # measure chunk size and overlap instead of guessing
```

## Design decisions

- **Markdown-aware chunking.** Splits at headings and records the heading path on every chunk (it is embedded
  with the text), never splits inside a code block, and carries overlap between chunks.
- **Qdrant is the single source of truth.** The BM25 index is rebuilt from Qdrant payloads, and the hash of each
  file is stored beside its chunks, so the keyword index and the vector index cannot disagree.
- **Incremental indexing.** A file is re-chunked only if its hash, the chunking parameters or the chunker version
  changed. Re-embedding is further limited by an embedding cache keyed on `sha256(model + text)`.
- **Reciprocal Rank Fusion** merges the two ranked lists using ranks only, because cosine similarity and BM25
  scores are on different scales and cannot be averaged meaningfully.
- **Prompt rules that matter more than fluency:** every claim cites a numbered passage, and the model must say the
  documentation does not answer rather than use outside knowledge.

## Deploying on Kubernetes

Manifests are in `k8s/`: a Qdrant StatefulSet with a persistent volume, a one-shot indexing Job, and the API
Deployment with startup, liveness and readiness probes, a rolling update that never drops capacity, and a
CPU-based HorizontalPodAutoscaler.

```bash
docker build -t docs-rag-assistant:latest .
kubectl apply -f k8s/config.yaml -f k8s/qdrant.yaml
kubectl apply -f k8s/indexer-job.yaml
kubectl apply -f k8s/api.yaml
```

**Status: the manifests, Dockerfile and GitHub Actions workflow are written but have not been run.** They were
authored on a machine without Docker, kubectl or Git. Treat them as untested until they have been applied to a
cluster and the workflow has executed once.

## CI

`.github/workflows/ci.yml` runs lint and unit tests on every pull request, then rebuilds the index and runs the
retrieval quality gate. A change that drops hit@5, MRR, precision or recall below `eval/thresholds.yaml` fails the
check. Model weights and computed embeddings are cached between runs.
