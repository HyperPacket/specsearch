from collections.abc import Iterator

import chromadb
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

from specsearch import config
from specsearch.ingest import Chunk

# ES index config — english analyzer gives stemming + stopword removal for spec text.
_ES_SETTINGS: dict = {"analysis": {"analyzer": {"default": {"type": "english"}}}}
_ES_MAPPINGS: dict = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "text": {"type": "text", "analyzer": "english"},
        "source_file": {"type": "keyword"},
        "page": {"type": "integer"},
        "section": {"type": "keyword"},
    }
}


def _es_docs(chunks: list[Chunk]) -> Iterator[dict]:
    for c in chunks:
        yield {
            "_index": config.ES_INDEX,
            "_id": c.chunk_id,
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_file": c.source_file,
            "page": c.page,
            "section": c.section,
        }


def build_es_index(chunks: list[Chunk]) -> int:
    """Create/recreate the ES index and bulk-index chunks. Returns doc count."""
    es = Elasticsearch(config.ES_URL)
    if es.indices.exists(index=config.ES_INDEX):
        es.indices.delete(index=config.ES_INDEX)
    es.indices.create(index=config.ES_INDEX, settings=_ES_SETTINGS, mappings=_ES_MAPPINGS)
    if chunks:
        bulk(es, _es_docs(chunks))
        es.indices.refresh(index=config.ES_INDEX)
    return len(chunks)


def build_vector_index(chunks: list[Chunk]) -> int:
    """Embed chunks and upsert into a persistent Chroma collection. Returns doc count."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if any(c.name == config.ES_INDEX for c in chroma.list_collections()):
        chroma.delete_collection(name=config.ES_INDEX)
    collection = chroma.create_collection(
        name=config.ES_INDEX,
        metadata={"hnsw:space": "cosine"},
    )
    if not chunks:
        return 0
    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        [c.text for c in chunks], batch_size=64, show_progress_bar=True
    )
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "page": c.page, "section": c.section}
            for c in chunks
        ],
    )
    return len(chunks)
