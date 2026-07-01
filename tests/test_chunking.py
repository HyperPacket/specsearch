from pathlib import Path

from specsearch.ingest import Chunk, _sliding_window, ingest_directory, ingest_pdf


def test_sliding_window_short_text() -> None:
    words = "hello world foo bar".split()
    assert _sliding_window(words, size=512, overlap=64) == ["hello world foo bar"]


def test_sliding_window_overlap() -> None:
    words = [str(i) for i in range(600)]
    windows = _sliding_window(words, size=512, overlap=64)
    assert len(windows) == 2
    # overlap region: last 64 words of window 0 == first 64 words of window 1
    w0, w1 = windows[0].split(), windows[1].split()
    assert w0[-64:] == w1[:64]


def test_sliding_window_no_empty_chunks() -> None:
    words = [str(i) for i in range(1000)]
    for window in _sliding_window(words, size=512, overlap=64):
        assert window.strip()


def test_ingest_pdf_returns_chunks(sample_spec_pdf: Path) -> None:
    chunks = ingest_pdf(sample_spec_pdf)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.text.strip()
        assert chunk.source_file == sample_spec_pdf.name
        assert chunk.page >= 1
        assert chunk.chunk_id


def test_ingest_pdf_detects_csi_section(sample_spec_pdf: Path) -> None:
    chunks = ingest_pdf(sample_spec_pdf)
    sections = {c.section for c in chunks}
    assert any("03 30 00" in s for s in sections), f"CSI section not detected; got: {sections}"
    assert any("03 45 00" in s for s in sections), f"Second CSI section missing; got: {sections}"


def test_chunk_ids_unique(sample_spec_pdf: Path) -> None:
    chunks = ingest_pdf(sample_spec_pdf)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids"


def test_ingest_directory(tmp_path: Path, sample_spec_pdf: Path) -> None:
    import shutil

    shutil.copy(sample_spec_pdf, tmp_path / "spec_a.pdf")
    shutil.copy(sample_spec_pdf, tmp_path / "spec_b.pdf")
    chunks = ingest_directory(tmp_path)
    sources = {c.source_file for c in chunks}
    assert sources == {"spec_a.pdf", "spec_b.pdf"}
