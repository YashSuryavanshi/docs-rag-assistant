"""Answer generation with citations, against any OpenAI-compatible endpoint.

vLLM's server speaks the OpenAI API, so the same code works with a local
`vllm serve <model>`, a hosted provider, or anything else exposing
/v1/chat/completions. Configure with RAG_LLM_BASE_URL and RAG_LLM_MODEL.

The prompt forces two behaviours that matter more than fluency:
- every claim cites a numbered passage, so answers can be checked;
- if the passages do not contain the answer, the model says so instead of
  answering from memory (the main source of RAG hallucinations).
"""

from __future__ import annotations

from openai import OpenAI

from rag.config import Settings
from rag.store import Hit

SYSTEM_PROMPT = """You answer questions about vLLM using ONLY the numbered passages provided.
Rules:
- Cite the passage for every claim, like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: "The documentation provided does not answer this."
- Do not use knowledge from outside the passages. Be concise."""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.source} | {h.heading_path})\n{h.text}" for i, h in enumerate(hits, start=1)
    )


class Generator:
    def __init__(self, settings: Settings):
        self.s = settings
        self.enabled = bool(settings.llm_base_url and settings.llm_model)
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key) if self.enabled else None

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("No LLM configured: set RAG_LLM_BASE_URL and RAG_LLM_MODEL")
        resp = self._client.chat.completions.create(
            model=self.s.llm_model,
            temperature=0.0,
            max_tokens=max_tokens or self.s.llm_max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    def answer(self, question: str, hits: list[Hit]) -> str | None:
        """Returns None when no LLM is configured; callers then show passages only."""
        if not self.enabled:
            return None
        return self.chat(SYSTEM_PROMPT, f"Passages:\n{format_context(hits)}\n\nQuestion: {question}")
