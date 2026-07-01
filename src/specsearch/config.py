import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ES_URL: str = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX: str = os.getenv("ES_INDEX", "specsearch")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", "data/chroma"))
RAW_DIR: Path = Path(os.getenv("RAW_DIR", "data/raw"))
TOP_K: int = int(os.getenv("TOP_K", "6"))
RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "1000"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
