
from specsearch.generate import grounded_citation_check
from specsearch.ingest import Chunk


def _chunk(source: str, page: int) -> Chunk:
    return Chunk(chunk_id=f"{source}:{page}", text="t", source_file=source, page=page, section="s")


def test_valid_citation_kept() -> None:
    chunks = [_chunk("spec.pdf", 14)]
    result = grounded_citation_check("Strength is 4000 psi [spec.pdf p.14].", chunks)
    assert "[spec.pdf p.14]" in result
    assert "UNVERIFIED" not in result


def test_invalid_citation_flagged() -> None:
    chunks = [_chunk("spec.pdf", 14)]
    result = grounded_citation_check("See [ghost.pdf p.99].", chunks)
    assert "[UNVERIFIED: ghost.pdf p.99]" in result
    assert "[ghost.pdf p.99]" not in result.replace("[UNVERIFIED: ghost.pdf p.99]", "")


def test_no_citations_unchanged() -> None:
    chunks = [_chunk("spec.pdf", 1)]
    answer = "No citations in this answer."
    assert grounded_citation_check(answer, chunks) == answer


def test_mixed_valid_and_invalid() -> None:
    chunks = [_chunk("spec.pdf", 5), _chunk("spec.pdf", 10)]
    answer = "Good [spec.pdf p.5] and bad [hallucinated.pdf p.1]."
    result = grounded_citation_check(answer, chunks)
    assert "[spec.pdf p.5]" in result
    assert "[UNVERIFIED: hallucinated.pdf p.1]" in result


def test_multiple_valid_citations() -> None:
    chunks = [_chunk("a.pdf", 1), _chunk("b.pdf", 2)]
    answer = "First [a.pdf p.1] second [b.pdf p.2]."
    result = grounded_citation_check(answer, chunks)
    assert "[a.pdf p.1]" in result
    assert "[b.pdf p.2]" in result
    assert "UNVERIFIED" not in result


def test_empty_chunks_flags_all() -> None:
    answer = "See [spec.pdf p.1] and [spec.pdf p.2]."
    result = grounded_citation_check(answer, [])
    assert result.count("UNVERIFIED") == 2


def test_filename_with_hyphens_and_dots() -> None:
    chunks = [_chunk("my-project-spec.v2.pdf", 7)]
    answer = "Load [my-project-spec.v2.pdf p.7] here."
    result = grounded_citation_check(answer, chunks)
    assert "[my-project-spec.v2.pdf p.7]" in result
    assert "UNVERIFIED" not in result


def test_wrong_page_flagged() -> None:
    chunks = [_chunk("spec.pdf", 14)]
    answer = "Wrong page [spec.pdf p.15]."
    result = grounded_citation_check(answer, chunks)
    assert "[UNVERIFIED: spec.pdf p.15]" in result
