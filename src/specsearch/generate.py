import re

import ollama

from specsearch import config
from specsearch.ingest import Chunk

# Matches [filename p.N] — lazy inner match stops before the last " p.N" token.
_CITE_RE = re.compile(r"\[([^\]\s][^\]]*?)\s+p\.(\d+)\]")

_SYSTEM = (
    "You are a construction specification expert. "
    "Answer using ONLY the provided excerpts. "
    "Cite every factual claim as [filename p.N] using the exact filename shown. "
    "If the answer is not found in the excerpts, say so explicitly."
)


def _build_context(chunks: list[Chunk]) -> str:
    parts = [
        f"[{i}] [{c.source_file} p.{c.page}] ({c.section})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)


def grounded_citation_check(answer: str, chunks: list[Chunk]) -> str:
    """Replace any [source p.N] citation not backed by a provided chunk with [UNVERIFIED: ...]."""
    valid = {(c.source_file, c.page) for c in chunks}

    def _check(m: re.Match) -> str:  # type: ignore[type-arg]
        source, page = m.group(1).strip(), int(m.group(2))
        return m.group(0) if (source, page) in valid else f"[UNVERIFIED: {source} p.{page}]"

    return _CITE_RE.sub(_check, answer)


def generate(query: str, chunks: list[Chunk]) -> str:
    """Call Ollama with top-k chunks as grounded context; return raw answer text."""
    context = _build_context(chunks)
    prompt = (
        f"SPECIFICATION EXCERPTS:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER (cite every claim as [filename p.N]):"
    )
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )
    return response.message.content


def generate_cited(query: str, chunks: list[Chunk]) -> str:
    """Generate answer then strip any hallucinated citations."""
    return grounded_citation_check(generate(query, chunks), chunks)


def parse_citations(text: str) -> list[str]:
    """Return unique citation strings found in text, in order of appearance."""
    return list(dict.fromkeys(m.group(0) for m in _CITE_RE.finditer(text)))
