from rag.chunking import chunk_markdown, strip_front_matter

DOC = """---
title: Example
---
# Prefix Caching

Intro paragraph about caching.

## Enabling

Set the flag.

--8<-- "examples/snippet.py"

```python
llm = LLM(model="m", enable_prefix_caching=True)

print("blank line above stays inside the fence")
```
"""


def test_front_matter_removed():
    assert "title: Example" not in strip_front_matter(DOC)


def test_heading_path_and_directive_stripped():
    chunks = chunk_markdown("features/apc.md", DOC, chunk_size=200, overlap=0)
    paths = {c.heading_path for c in chunks}
    assert "Prefix Caching" in paths
    assert "Prefix Caching > Enabling" in paths
    assert all("--8<--" not in c.text for c in chunks)


def test_code_block_never_split():
    # A tiny chunk size forces splits; the fenced block must still stay whole.
    chunks = chunk_markdown("features/apc.md", DOC, chunk_size=12, overlap=0)
    with_code = [c for c in chunks if "```python" in c.text]
    assert len(with_code) == 1
    assert with_code[0].text.count("```") == 2


def test_overlap_carries_previous_words():
    text = "# T\n\n" + "\n\n".join(f"para{i} " + "word " * 30 for i in range(5))
    chunks = chunk_markdown("t.md", text, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    tail = chunks[0].text.split()[-10:]
    assert chunks[1].text.split()[:10] == tail


def test_overlap_must_be_smaller_than_size():
    try:
        chunk_markdown("t.md", "# T\n\ntext", chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
