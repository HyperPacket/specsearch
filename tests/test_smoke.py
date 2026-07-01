"""
Integration smoke tests — auto-skip if Elasticsearch or Ollama are not running.
Run after: docker compose up -d elasticsearch && uv run specsearch index data/raw

Uses a dedicated ES index ("specsearch_smoke") and a temp Chroma dir so these
tests never overwrite the production index built by `make index`.
"""
import tempfile
import urllib.request
from pathlib import Path

import pytest

import specsearch.config as _cfg

_SMOKE_ES_INDEX = "specsearch_smoke"


def _up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except Exception:
        return False


_ES_UP = _up("http://localhost:9200")
_OLLAMA_UP = _up("http://localhost:11434")

needs_es = pytest.mark.skipif(not _ES_UP, reason="Elasticsearch not running")
needs_ollama = pytest.mark.skipif(not _OLLAMA_UP, reason="Ollama not running")


@pytest.fixture(scope="session")
def indexed(sample_spec_pdf: Path):
    """Index the fixture PDF into isolated smoke-test indexes, never the production ones."""
    if not _ES_UP:
        pytest.skip("Elasticsearch not running")
    from specsearch.index import build_es_index, build_vector_index
    from specsearch.ingest import ingest_pdf

    original_es = _cfg.ES_INDEX
    original_chroma = _cfg.CHROMA_DIR

    with tempfile.TemporaryDirectory() as tmp_chroma:
        _cfg.ES_INDEX = _SMOKE_ES_INDEX
        _cfg.CHROMA_DIR = Path(tmp_chroma)
        try:
            chunks = ingest_pdf(sample_spec_pdf)
            build_es_index(chunks)
            build_vector_index(chunks)
            yield chunks
        finally:
            _cfg.ES_INDEX = original_es
            _cfg.CHROMA_DIR = original_chroma


@needs_es
def test_es_index_builds(indexed: list) -> None:
    assert len(indexed) > 0


@needs_es
def test_bm25_retrieval(indexed: list) -> None:
    from specsearch.retrieve import retrieve_bm25

    results = retrieve_bm25("concrete compressive strength 4000 psi", top_n=10)
    assert len(results) > 0
    assert any("concrete" in c.text.lower() for c in results)


@needs_es
def test_vector_retrieval(indexed: list) -> None:
    from specsearch.retrieve import retrieve_vector

    results = retrieve_vector("concrete strength requirements", top_n=5)
    assert len(results) > 0


@needs_es
def test_retrieve_and_fuse(indexed: list) -> None:
    from specsearch.retrieve import retrieve_and_fuse

    results = retrieve_and_fuse("concrete compressive strength")
    assert len(results) > 0
    ids = [c.chunk_id for c in results]
    assert len(ids) == len(set(ids)), "RRF output has duplicate chunk_ids"


@needs_es
def test_rerank_pipeline(indexed: list) -> None:
    from specsearch.rerank import rerank
    from specsearch.retrieve import retrieve_and_fuse

    candidates = retrieve_and_fuse("concrete 4000 psi 28 days")
    top = rerank("concrete 4000 psi 28 days", candidates)
    assert 0 < len(top) <= 6


@needs_es
@needs_ollama
def test_full_pipeline(indexed: list) -> None:
    from specsearch.pipeline import run

    answer = run("What is the concrete compressive strength at 28 days?")
    assert answer.text.strip()
    assert answer.timings["total"] > 0
    assert "retrieve_fuse" in answer.timings


@needs_es
@needs_ollama
def test_explain_mode(indexed: list) -> None:
    from specsearch.pipeline import run

    answer = run("What is the concrete compressive strength?", explain=True)
    assert answer.candidates
    assert answer.rerank_scores
    assert len(answer.rerank_scores) <= 6
