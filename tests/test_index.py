from specsearch.index import _ES_MAPPINGS, _ES_SETTINGS


def test_es_uses_english_analyzer() -> None:
    analyzer = _ES_SETTINGS["analysis"]["analyzer"]["default"]
    assert analyzer["type"] == "english"


def test_es_mapping_fields() -> None:
    props = _ES_MAPPINGS["properties"]
    assert props["text"]["analyzer"] == "english"
    assert props["chunk_id"]["type"] == "keyword"
    assert props["page"]["type"] == "integer"


def test_es_mapping_covers_chunk_fields() -> None:
    props = _ES_MAPPINGS["properties"]
    for field in ("chunk_id", "text", "source_file", "page", "section"):
        assert field in props, f"Missing field: {field}"
