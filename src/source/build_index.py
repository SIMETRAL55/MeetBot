from pathlib import Path
import json
import logging
import os
import hashlib
from typing import List, Dict, Any, Optional
from config import settings
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _file_hash(path: Path, model_name: str) -> str:
    """Compute sha256 of file contents + model name (so different models produce different hashes)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    h.update(model_name.encode("utf-8"))
    return h.hexdigest()


def _index_meta_path(persist_dir: Path) -> Path:
    return persist_dir / ".index_meta.json"


def _read_index_meta(persist_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = _index_meta_path(persist_dir)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read index meta %s: %s", meta_path, e)
        return None


def _write_index_meta(persist_dir: Path, meta: Dict[str, Any]) -> None:
    meta_path = _index_meta_path(persist_dir)
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_prepared_jsonl(prepared_path: str) -> List[Dict[str, Any]]:
    p = Path(prepared_path)
    if not p.exists():
        raise FileNotFoundError(f"prepared file not found: {prepared_path}")

    docs = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {prepared_path}: {e.msg}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {lineno} of {prepared_path}, got {type(obj)}")
            docs.append(obj)
    return docs


def create_chroma_index(prepared_jsonl: str,
                        persist_root: str = "db",
                        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                        huggingface_token: Optional[str] = None,
                        collection_name: Optional[str] = None):
    """
    Build or update a Chroma collection from prepared JSONL.
    If an existing index metadata file exists and matches the prepared file hash + model_name,
    the function will skip re-computing embeddings and load the existing index.

    Returns:
      - for langchain_community path: the Chroma vectorstore object
      - for fallback chromadb path: a dict {"chroma_client": client, "collection": collection}
    """
    docs = _load_prepared_jsonl(prepared_jsonl)
    if len(docs) == 0:
        logger.warning("No docs found in prepared file: %s", prepared_jsonl)
        return None

    # ensure token available via env (new HF embedding classes read from env)
    hf_token = huggingface_token or getattr(settings, "HF_API_TOKEN", None)
    if hf_token:
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)

    audio_basename = Path(prepared_jsonl).stem
    persist_dir = Path(persist_root) / audio_basename
    persist_dir.mkdir(parents=True, exist_ok=True)

    coll_name = collection_name or audio_basename
    logger.info("Using embedding model: %s", model_name)

    # compute current prepared file hash
    prepared_path = Path(prepared_jsonl)
    current_hash = _file_hash(prepared_path, model_name)

    # If .index_meta.json exists and hashes match -> skip embedding
    existing_meta = _read_index_meta(persist_dir)
    if existing_meta and existing_meta.get("hash") == current_hash:
        logger.info("Index already up-to-date for %s (model=%s). Skipping embedding.", audio_basename, model_name)
        # Try to load and return the existing index in the same API used below
        try:
            emb = None
            try:
                emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
            except Exception:
                emb = None

            try:
                if emb is not None:
                    vectordb = Chroma(persist_directory=str(persist_dir), collection_name=coll_name, embedding_function=emb)
                else:
                    vectordb = Chroma(persist_directory=str(persist_dir), collection_name=coll_name)
                logger.info("Loaded existing Chroma vectorstore from %s", persist_dir)
                return vectordb
            except Exception as e:
                logger.warning("Failed to load langchain_community Chroma: %s", e)
        except Exception:
            # langchain_community not available, try chromadb fallback
            pass

        # fallback: load chromadb client & collection
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.Client(ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_dir)))
            if client.exists(coll_name):
                collection = client.get_collection(coll_name)
                logger.info("Loaded existing chromadb collection %s from %s", coll_name, persist_dir)
                return {"chroma_client": client, "collection": collection}
            else:
                logger.warning("Meta matched but collection %s not found in chromadb at %s", coll_name, persist_dir)
        except Exception as e:
            logger.warning("Failed to load chromadb fallback: %s", e)

    try:


        emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})  # reads token from env
        lc_docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in docs]

        logger.info("Persisting Chroma DB to %s (collection=%s) via langchain_community", persist_dir, coll_name)
        vectordb = Chroma.from_documents(documents=lc_docs,
                                         embedding=emb,
                                         persist_directory=str(persist_dir),
                                         collection_name=coll_name)
        vectordb.persist()

        # write index meta
        meta = {"hash": current_hash, "model": model_name, "n_docs": len(lc_docs)}
        _write_index_meta(persist_dir, meta)
        logger.info("Chroma index built with %d docs (langchain_community)", len(lc_docs))
        return vectordb

    except Exception as e:
        logger.warning("langchain_community path failed or not installed: %s", e)
        logger.info("Falling back to sentence-transformers + chromadb (direct)")

    # --- Fallback: sentence-transformers + chromadb ---
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except Exception as e:
        raise RuntimeError("Fallback dependencies missing. Install either langchain_community or sentence-transformers + chromadb.") from e

    # Build embeddings with sentence-transformers
    sbert = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]
    embs = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Create chromadb client (persist to directory)
    client = chromadb.Client(ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_dir)))
    # create or get collection
    if client.exists(coll_name):
        collection = client.get_collection(coll_name)
        # optionally: update (upsert) changed docs
        logger.info("Updating existing collection %s", coll_name)
    else:
        collection = client.create_collection(name=coll_name)

    ids = [d["id"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    collection.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=embs.tolist())
    client.persist()

    # write index meta
    meta = {"hash": current_hash, "model": model_name, "n_docs": len(docs)}
    _write_index_meta(persist_dir, meta)
    logger.info("Chroma (direct) index built with %d docs", len(docs))
    return {"chroma_client": client, "collection": collection}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_jsonl", help="Path to prepared/<audio>.jsonl")
    parser.add_argument("--persist-root", default="db")
    parser.add_argument("--model", default="sbintuitions/sarashina-embedding-v1-1b")
    parser.add_argument("--token", default=None, help="Hugging Face token (optional; will use config)")
    args = parser.parse_args()
    create_chroma_index(args.prepared_jsonl, persist_root=args.persist_root, model_name=args.model, huggingface_token=args.token)
