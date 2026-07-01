from unittest.mock import patch

import numpy as np

from specsearch import config
from specsearch.ingest import Chunk
from specsearch.rerank import rerank, rerank_with_scores


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, text="x", source_file="f.pdf", page=1, section="s")


def _mock_model(scores: list[float]):
    """Patch _get_rerank_model to return a mock that predicts the given scores."""
    return patch(
        "specsearch.rerank._get_rerank_model",
        return_value=type("M", (), {"predict": lambda self, pairs: np.array(scores)})(),
    )


def test_rerank_empty_candidates() -> None:
    assert rerank("q", []) == []


def test_rerank_with_scores_empty_candidates() -> None:
    assert rerank_with_scores("q", []) == []


def test_rerank_sorts_descending() -> None:
    chunks = [_chunk("low"), _chunk("high"), _chunk("mid")]
    with _mock_model([1.0, 3.0, 2.0]):
        result = rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in result] == ["high", "mid", "low"]


def test_rerank_slices_to_top_k() -> None:
    chunks = [_chunk(f"id{i}") for i in range(10)]
    with _mock_model(list(range(10, 0, -1))):
        result = rerank("q", chunks, top_k=3)
    assert len(result) == 3
    assert result[0].chunk_id == "id0"


def test_rerank_caps_at_rerank_candidates() -> None:
    chunks = [_chunk(f"id{i}") for i in range(config.RERANK_CANDIDATES + 200)]
    seen: list[int] = []

    def fake_predict(self, pairs: list) -> np.ndarray:
        seen.append(len(pairs))
        return np.zeros(len(pairs))

    with patch(
        "specsearch.rerank._get_rerank_model",
        return_value=type("M", (), {"predict": fake_predict})(),
    ):
        rerank("q", chunks)

    assert seen[0] == config.RERANK_CANDIDATES


def test_rerank_with_scores_returns_float_chunk_pairs() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    with _mock_model([0.9, 0.1]):
        result = rerank_with_scores("q", chunks, top_k=2)
    assert len(result) == 2
    scores, items = zip(*result)
    assert scores[0] > scores[1]
    assert items[0].chunk_id == "a"
