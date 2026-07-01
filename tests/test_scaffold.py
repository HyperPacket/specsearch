from pathlib import Path

from specsearch import config


def test_config_defaults() -> None:
    """Defaults match the values documented in .env.example."""
    assert config.TOP_K == 6
    assert config.CHUNK_SIZE == 512
    assert config.CHUNK_OVERLAP == 64


def test_config_types() -> None:
    """Config module must coerce env vars into the correct Python types."""
    assert isinstance(config.TOP_K, int)
    assert isinstance(config.CHUNK_SIZE, int)
    assert isinstance(config.CHUNK_OVERLAP, int)
    assert isinstance(config.CHROMA_DIR, Path)
    assert isinstance(config.ES_URL, str)
    assert config.ES_URL.startswith("http")


def test_chunk_overlap_less_than_size() -> None:
    """Sliding window step = size - overlap; overlap >= size causes an infinite loop."""
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
