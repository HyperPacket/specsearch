from concurrent.futures import ThreadPoolExecutor

import chromadb
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from specsearch import config
from specsearch.ingest import Chunk

# module-level singleton — loaded once per process; use DI if multi-process needed
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
    return _embed_model


def retrieve_bm25(query: str, top_n: int = 500) -> list[Chunk]:
    """BM25 retrieval via Elasticsearch. Returns up to top_n ranked chunks."""
    es = Elasticsearch(config.ES_URL)
    resp = es.search(
        index=config.ES_INDEX,
        query={"match": {"text": query}},
        size=top_n,
    )
    return [
        Chunk(
            chunk_id=hit["_source"]["chunk_id"],
            text=hit["_source"]["text"],
            source_file=hit["_source"]["source_file"],
            page=int(hit["_source"]["page"]),
            section=hit["_source"]["section"],
        )
        for hit in resp["hits"]["hits"]
    ]


def retrieve_vector(query: str, top_n: int = 500) -> list[Chunk]:
    """Dense vector retrieval via ChromaDB (cosine). Returns up to top_n ranked chunks."""
    query_vec = _get_embed_model().encode(query).tolist()
    chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma.get_collection(name=config.ES_INDEX)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_n, collection.count()),
        include=["documents", "metadatas"],
    )
    chunks: list[Chunk] = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        chunks.append(
            Chunk(
                chunk_id=doc_id,
                text=results["documents"][0][i],
                source_file=meta["source_file"],
                page=int(meta["page"]),
                section=meta["section"],
            )
        )
    return chunks


def rrf_fuse(ranked_lists: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank). Merges by chunk_id."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk.chunk_id] = chunk
    return [chunks[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)]


def parallel_retrieve(query: str, top_n: int = 500) -> tuple[list[Chunk], list[Chunk]]:
    """Run BM25 and vector retrieval concurrently. Returns (bm25, vector) result lists."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        bm25_fut = ex.submit(retrieve_bm25, query, top_n)
        vec_fut = ex.submit(retrieve_vector, query, top_n)
        return bm25_fut.result(), vec_fut.result()


def retrieve_and_fuse(query: str, top_n: int = 500) -> list[Chunk]:
    """Parallel BM25 + vector retrieve → RRF fusion (k=60). Returns merged candidates."""
    bm25, vector = parallel_retrieve(query, top_n)
    return rrf_fuse([bm25, vector])
