from specsearch.eval import mrr, ndcg_at_k, recall_at_k
from specsearch.ingest import Chunk


def _chunk(source: str, page: int) -> Chunk:
    return Chunk(chunk_id=f"{source}:{page}", text="t", source_file=source, page=page, section="s")


_REL = [{"source_file": "spec.pdf", "page": 3}]


def test_recall_hit() -> None:
    ranked = [_chunk("spec.pdf", 3), _chunk("other.pdf", 1)]
    assert recall_at_k(ranked, _REL, k=10) == 1.0


def test_recall_miss() -> None:
    ranked = [_chunk("other.pdf", 1)]
    assert recall_at_k(ranked, _REL, k=10) == 0.0


def test_recall_outside_k() -> None:
    ranked = [_chunk("other.pdf", i) for i in range(10)] + [_chunk("spec.pdf", 3)]
    assert recall_at_k(ranked, _REL, k=10) == 0.0


def test_mrr_first() -> None:
    ranked = [_chunk("spec.pdf", 3)]
    assert mrr(ranked, _REL) == 1.0


def test_mrr_second() -> None:
    ranked = [_chunk("other.pdf", 1), _chunk("spec.pdf", 3)]
    assert mrr(ranked, _REL) == 0.5


def test_mrr_miss() -> None:
    assert mrr([_chunk("x.pdf", 9)], _REL) == 0.0


def test_ndcg_perfect() -> None:
    ranked = [_chunk("spec.pdf", 3)]
    assert ndcg_at_k(ranked, _REL) == 1.0


def test_ndcg_miss() -> None:
    assert ndcg_at_k([_chunk("x.pdf", 1)], _REL) == 0.0


def test_ndcg_rank_2() -> None:
    import math
    ranked = [_chunk("x.pdf", 1), _chunk("spec.pdf", 3)]
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0
    expected = (1.0 / math.log2(3)) / 1.0
    assert abs(ndcg_at_k(ranked, _REL) - expected) < 1e-9


def test_empty_relevant() -> None:
    ranked = [_chunk("spec.pdf", 1)]
    assert recall_at_k(ranked, [], k=10) == 0.0
    assert mrr(ranked, []) == 0.0
    assert ndcg_at_k(ranked, []) == 0.0
