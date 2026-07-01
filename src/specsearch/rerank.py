import numpy as np
from sentence_transformers import CrossEncoder

from specsearch import config
from specsearch.ingest import Chunk

# module-level singleton — ~90 MB model, load once per process
_rerank_model: CrossEncoder | None = None


def _get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(config.RERANK_MODEL)
    return _rerank_model


def rerank(query: str, candidates: list[Chunk], top_k: int | None = None) -> list[Chunk]:
    """Score query–chunk pairs with the cross-encoder; return top_k by score.

    Caps input at RERANK_CANDIDATES (default 1000) before scoring — feasible on GPU,
    lower if running on CPU (set RERANK_CANDIDATES=200 in .env).
    """
    if not candidates:
        return []
    k = top_k if top_k is not None else config.TOP_K
    cap = min(len(candidates), config.RERANK_CANDIDATES)
    pool = candidates[:cap]
    scores: np.ndarray = _get_rerank_model().predict([(query, c.text) for c in pool])
    ranked = sorted(zip(scores.tolist(), pool), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:k]]


def rerank_with_scores(
    query: str, candidates: list[Chunk], top_k: int | None = None
) -> list[tuple[float, Chunk]]:
    """Same as rerank() but returns (score, chunk) pairs — used by --explain."""
    if not candidates:
        return []
    k = top_k if top_k is not None else config.TOP_K
    cap = min(len(candidates), config.RERANK_CANDIDATES)
    pool = candidates[:cap]
    scores: np.ndarray = _get_rerank_model().predict([(query, c.text) for c in pool])
    ranked = sorted(zip(scores.tolist(), pool), key=lambda x: x[0], reverse=True)
    return ranked[:k]
