from specsearch.ingest import Chunk
from specsearch.retrieve import rrf_fuse


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, text="x", source_file="f.pdf", page=1, section="s")


def test_rrf_single_list_preserves_order() -> None:
    chunks = [_chunk(f"id{i}") for i in range(5)]
    result = rrf_fuse([chunks])
    assert [c.chunk_id for c in result] == [c.chunk_id for c in chunks]


def test_rrf_overlap_ranks_first() -> None:
    # A doc appearing in both lists scores higher than one in only one list.
    shared = _chunk("shared")
    only_a = _chunk("only_a")
    only_b = _chunk("only_b")
    result = rrf_fuse([[shared, only_a], [shared, only_b]])
    assert result[0].chunk_id == "shared"


def test_rrf_empty_lists() -> None:
    assert rrf_fuse([[], []]) == []


def test_rrf_one_empty_one_full() -> None:
    chunks = [_chunk(f"id{i}") for i in range(3)]
    result = rrf_fuse([chunks, []])
    assert len(result) == 3


def test_rrf_score_formula() -> None:
    # With k=60, rank-1 doc score = 1/61; rank-2 score = 1/62.
    # Doc in rank-1 of list A must outscore doc in rank-2 of list A when list B is empty.
    a = _chunk("a")
    b = _chunk("b")
    result = rrf_fuse([[a, b]], k=60)
    assert result[0].chunk_id == "a"
    assert result[1].chunk_id == "b"


def test_rrf_deduplicates_by_chunk_id() -> None:
    chunk = _chunk("dup")
    result = rrf_fuse([[chunk, chunk]])
    assert len(result) == 1


def test_rrf_k_affects_score_spread() -> None:
    # Higher k → scores are more compressed (differences smaller).
    # Doc at rank 1 with k=1: score=0.5; with k=1000: score≈0.001.
    # Ordering should still be preserved.
    chunks = [_chunk(f"id{i}") for i in range(10)]
    r1 = rrf_fuse([chunks], k=1)
    r1000 = rrf_fuse([chunks], k=1000)
    assert [c.chunk_id for c in r1] == [c.chunk_id for c in r1000]
