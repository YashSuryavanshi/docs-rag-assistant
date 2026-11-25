"""Download the vLLM documentation as the RAG corpus.

Pins the corpus to one commit so evaluation results are reproducible: the
commit SHA is written to data/corpus/MANIFEST.json, and the eval set refers
to files by their path inside that snapshot.

Usage:
    python scripts/fetch_corpus.py --ref pinned # the commit in MANIFEST.json (CI, Kubernetes)
    python scripts/fetch_corpus.py              # latest main (then re-check eval labels)
    python scripts/fetch_corpus.py --ref <sha>  # a specific commit
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = "vllm-project/vllm"
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus" / "vllm"
MANIFEST = ROOT / "data" / "corpus" / "MANIFEST.json"


def resolve_sha(ref: str) -> str:
    if ref == "pinned":
        return json.loads(MANIFEST.read_text())["sha"]
    url = f"https://api.github.com/repos/{REPO}/commits/{ref}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["sha"]


def download_docs(sha: str) -> int:
    url = f"https://codeload.github.com/{REPO}/zip/{sha}"
    with urllib.request.urlopen(url, timeout=600) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))

    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True)

    count = 0
    for name in archive.namelist():
        # Entries look like "vllm-<sha>/docs/serving/foo.md"; keep only docs/**/*.md.
        parts = name.split("/", 1)
        if len(parts) < 2 or not parts[1].startswith("docs/") or not parts[1].endswith(".md"):
            continue
        rel = Path(parts[1]).relative_to("docs")
        target = CORPUS_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="main", help="branch, tag or commit SHA")
    args = parser.parse_args()

    sha = resolve_sha(args.ref)
    count = download_docs(sha)
    MANIFEST.write_text(json.dumps({"repo": REPO, "sha": sha, "files": count}, indent=2))
    print(f"Fetched {count} markdown files from {REPO}@{sha[:12]} into {CORPUS_DIR}")


if __name__ == "__main__":
    main()
