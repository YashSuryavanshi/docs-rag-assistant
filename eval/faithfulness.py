"""Answer faithfulness, judged by an LLM.

Retrieval metrics say whether the right passages were found. Faithfulness says
whether the generated answer actually sticks to them. The judge splits the
answer into atomic claims and marks each as supported or not by the passages:

    faithfulness = supported claims / total claims

Needs an LLM endpoint (RAG_LLM_BASE_URL / RAG_LLM_MODEL), so it is optional
and not part of the CI gate, which must run without secrets.
"""

from __future__ import annotations

import json
import re

from rag.generate import Generator, format_context
from rag.store import Hit

JUDGE_PROMPT = """You check whether an answer is supported by source passages.
Split the ANSWER into short atomic factual claims. For each claim decide if the PASSAGES support it.
Reply with JSON only, in this shape:
{"claims": [{"claim": "...", "supported": true}]}"""


def faithfulness(generator: Generator, answer: str, hits: list[Hit]) -> float | None:
    if "does not answer this" in answer:
        return None  # an abstention makes no claims to check
    raw = generator.chat(JUDGE_PROMPT, f"PASSAGES:\n{format_context(hits)}\n\nANSWER:\n{answer}", max_tokens=800)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        claims = json.loads(match.group(0)).get("claims", [])
    except json.JSONDecodeError:
        return None
    if not claims:
        return None
    return sum(1 for c in claims if c.get("supported")) / len(claims)
