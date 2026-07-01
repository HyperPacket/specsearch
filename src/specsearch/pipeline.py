import time
from dataclasses import dataclass, field

from specsearch.generate import generate_cited, parse_citations
from specsearch.ingest import Chunk
from specsearch.rerank import rerank_with_scores
from specsearch.retrieve import retrieve_and_fuse


@dataclass
class Answer:
    text: str
    citations: list[str]
    timings: dict[str, float]
    candidates: list[Chunk] = field(default_factory=list)       # populated when explain=True
    rerank_scores: list[tuple[float, Chunk]] = field(default_factory=list)  # explain=True


def run(query: str, explain: bool = False) -> Answer:
    """Execute all six pipeline stages; return Answer with text, citations, and timings."""
    t0 = time.perf_counter()

    # Stages 3+4: parallel BM25 + vector retrieve → RRF fusion
    t1 = time.perf_counter()
    candidates = retrieve_and_fuse(query)
    t2 = time.perf_counter()

    # Stage 5: cross-encoder rerank (always use _with_scores so explain is free)
    scored = rerank_with_scores(query, candidates)
    top_chunks = [c for _, c in scored]
    t3 = time.perf_counter()

    # Stage 6: Ollama cited answer + grounded-citation check
    text = generate_cited(query, top_chunks)
    t4 = time.perf_counter()

    return Answer(
        text=text,
        citations=parse_citations(text),
        timings={
            "retrieve_fuse": round(t2 - t1, 3),
            "rerank": round(t3 - t2, 3),
            "generate": round(t4 - t3, 3),
            "total": round(t4 - t0, 3),
        },
        candidates=candidates if explain else [],
        rerank_scores=scored if explain else [],
    )
