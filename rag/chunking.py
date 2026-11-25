"""Markdown-aware chunking.

Why not split every N characters? Because retrieval quality depends on chunks
being coherent units of meaning. This chunker:

1. Splits a document into sections at headings, and records the heading path
   ("Automatic Prefix Caching > Enabling APC") on every chunk. The heading path
   is prepended to the text that gets embedded, so a chunk that only says
   "set enable_prefix_caching=True" still carries what it is about.
2. Never splits inside a fenced code block. Half a code sample is useless.
3. Packs whole paragraphs into a chunk up to `chunk_size` words, and carries
   `overlap` words from the previous chunk so that a sentence cut at a boundary
   still appears whole in one of the two chunks.

Sizes are measured in words, a cheap proxy for tokens (~1.3 tokens per word).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bump whenever chunking logic changes: it is part of every file's index hash,
# so a bump forces all files to be re-chunked on the next incremental index.
CHUNKER_VERSION = 2

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
# mkdocs "snippets" include directives carry no content of their own.
DIRECTIVE = re.compile(r"--8<--")


@dataclass
class Chunk:
    source: str  # path of the file inside the corpus, e.g. "features/lora.md"
    index: int  # position of the chunk within its file
    headings: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def heading_path(self) -> str:
        return " > ".join(self.headings)

    @property
    def embed_text(self) -> str:
        """Text that is embedded and indexed: heading path plus body."""
        return f"{self.heading_path}\n{self.text}" if self.headings else self.text


def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :]
    return text


def _sections(text: str) -> list[tuple[list[str], list[str]]]:
    """Split markdown into (heading_path, blocks) sections.

    A block is a paragraph, list, admonition or a whole fenced code block.
    """
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush_block() -> None:
        if current and any(line.strip() for line in current):
            blocks.append("\n".join(current).strip("\n"))
        current.clear()

    def flush_section() -> None:
        flush_block()
        if blocks:
            sections.append(([title for _, title in stack], blocks.copy()))
        blocks.clear()

    for line in text.splitlines():
        if DIRECTIVE.search(line):
            continue
        if FENCE.match(line):
            if not in_fence:
                flush_block()
            current.append(line)
            in_fence = not in_fence
            if not in_fence:
                flush_block()  # a closed code block is one block
            continue
        if in_fence:
            current.append(line)
            continue

        heading = HEADING.match(line)
        if heading:
            flush_section()
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        if not line.strip():
            flush_block()
        else:
            current.append(line)

    flush_section()
    return sections


def _split_long_block(block: str, size: int) -> list[str]:
    """A single block longer than `size` words is cut into word windows."""
    words = block.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def _pack(pieces: list[str], chunk_size: int) -> list[str]:
    """Greedily group whole pieces into bodies of at most `chunk_size` words."""
    bodies: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for piece in pieces:
        n = len(piece.split())
        if buf and buf_words + n > chunk_size:
            bodies.append("\n\n".join(buf))
            buf, buf_words = [], 0
        buf.append(piece)
        buf_words += n
    if buf:
        bodies.append("\n\n".join(buf))
    return bodies


def chunk_markdown(source: str, text: str, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for headings, blocks in _sections(strip_front_matter(text)):
        pieces: list[str] = []
        for block in blocks:
            is_code = FENCE.match(block) is not None
            if len(block.split()) > chunk_size and not is_code:
                pieces.extend(_split_long_block(block, chunk_size))
            else:
                # Oversized code blocks stay whole: a broken code sample is worse
                # than an oversized chunk (the embedder truncates past 512 tokens).
                pieces.append(block)

        carry = ""  # overlap words from the previous chunk in this section
        for body in _pack(pieces, chunk_size):
            text_out = f"{carry}\n\n{body}" if carry else body
            chunks.append(Chunk(source=source, index=len(chunks), headings=list(headings), text=text_out))
            carry = " ".join(body.split()[-overlap:]) if overlap else ""

    return chunks
