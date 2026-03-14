"""Vector indexing service for RAG document storage.

.. deprecated::
    This module is superseded by ``services.rag.indexer.RAGIndexer`` which
    provides atomic-swap indexing and speaker-aware chunking.  No production
    code paths import this module any longer.  It is retained only for
    reference and will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "meetbot.services.indexer is deprecated — use meetbot.services.rag.indexer instead",
    DeprecationWarning,
    stacklevel=2,
)

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

# Progress callback type: (stage: str, progress: float 0-100, message: str) -> None
ProgressCallback = Callable[[str, float, str], None]


class IndexerService:
    """
    Build and manage vector indices for RAG retrieval.

    Supports both LangChain Chroma and direct chromadb APIs with
    automatic fallback and index metadata tracking.
    """

    def __init__(self):
        """Initialize indexer service."""
        pass

    @staticmethod
    def _compute_hash(prepared_jsonl: Path, model_name: str) -> str:
        """
        Compute SHA256 hash of prepared JSONL + model name.

        Hash is used to detect if index needs rebuilding when
        prepared file or embedding model changes.

        Args:
            prepared_jsonl: Path to prepared JSONL file
            model_name: Embedding model name/path

        Returns:
            SHA256 hex digest
        """
        hasher = hashlib.sha256()

        # Hash file contents
        with prepared_jsonl.open("rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)

        # Hash model name to detect model changes
        hasher.update(model_name.encode("utf-8"))

        return hasher.hexdigest()

    @staticmethod
    def _get_metadata_path(persist_dir: Path) -> Path:
        """Get path to index metadata file."""
        return persist_dir / ".index_meta.json"

    @staticmethod
    def _read_metadata(persist_dir: Path) -> Optional[Dict[str, Any]]:
        """Read index metadata if it exists."""
        meta_path = IndexerService._get_metadata_path(persist_dir)
        if not meta_path.exists():
            return None

        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read metadata from {meta_path}: {e}")
            return None

    @staticmethod
    def _write_metadata(persist_dir: Path, meta: Dict[str, Any]) -> None:
        """Write index metadata."""
        meta_path = IndexerService._get_metadata_path(persist_dir)
        meta["created_at"] = datetime.now(timezone.utc).isoformat()

        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _load_prepared_jsonl(prepared_jsonl: Path) -> List[Dict[str, Any]]:
        """Load prepared JSONL documents."""
        docs = []
        with prepared_jsonl.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    docs.append(obj)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {prepared_jsonl}: {e}"
                    ) from e

        return docs

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """
        Resolve embedding model path.

        If local directory exists, use it. Otherwise, treat as HuggingFace ID.

        Args:
            model_name: Model name (could be local path or HF ID)

        Returns:
            Resolved model path or HF ID
        """
        local_path = Path(model_name)

        # Check if already a local directory path
        if local_path.exists() and local_path.is_dir():
            logger.info(f"Using local embedding model: {local_path}")
            return str(local_path)

        # Try to resolve from HF ID (e.g., "sbintuitions/sarashina-embedding-v1-1b")
        if "/" in model_name:
            model_dir = model_name.split("/")[-1]
            local_path = Path(model_dir)
            if local_path.exists() and local_path.is_dir():
                logger.info(f"Using local embedding model: {local_path}")
                return str(local_path)

            # Not found locally, use HF ID
            logger.info(f"Model not local, will download from HuggingFace: {model_name}")
            return model_name

        # Unknown format, assume HF model name
        return model_name

    def build_index(
        self,
        prepared_jsonl: str,
        persist_root: str = "db",
        embedding_model: str = "./models/all-MiniLM-L6-v2",
        collection_name: Optional[str] = None,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        device: Optional[str] = None,
    ) -> Any:
        """
        Build or load vector index from prepared documents.

        If index exists and hash matches, loads existing index.
        Otherwise, creates new index with embeddings.

        Args:
            prepared_jsonl: Path to prepared JSONL file
            persist_root: Root directory for vector database
            embedding_model: Embedding model name or path
            collection_name: Chroma collection name (defaults to audio basename)
            overwrite: Force rebuild even if index exists
            progress_callback: Optional callback for progress updates (stage, progress 0-100, message)

        Returns:
            Chroma vectorstore or chromadb collection object

        Raises:
            FileNotFoundError: If prepared JSONL not found
            RuntimeError: If index creation fails
        """
        # Resolve the compute device: explicit param > settings > cpu
        if device is None:
            try:
                from ..config import settings as _cfg
                device = _cfg.EMBEDDING_DEVICE
            except Exception:
                device = "cpu"
        device = device.lower().strip()
        logger.info(f"Embedding device: {device}")

        if progress_callback:
            progress_callback("indexing", 5, "Initializing indexing service...")

        prepared_path = Path(prepared_jsonl)
        if not prepared_path.exists():
            raise FileNotFoundError(f"Prepared JSONL not found: {prepared_jsonl}")

        if collection_name is None:
            collection_name = prepared_path.stem

        # Create a subdirectory for each collection/audio file
        # e.g., db/doctor_3/ for collection_name="doctor_3"
        persist_dir = Path(persist_root) / collection_name

        resolved_model = self._resolve_model_path(embedding_model)

        if progress_callback:
            progress_callback("indexing", 10, "Computing document hash...")
        # Compute hash for change detection
        current_hash = self._compute_hash(prepared_path, resolved_model)

        # Check if existing index matches
        if not overwrite:
            if progress_callback:
                progress_callback("indexing", 15, "Checking for existing index...")
            meta = self._read_metadata(persist_dir)
            if meta and meta.get("hash") == current_hash:
                if progress_callback:
                    progress_callback("indexing", 20, "Loading existing index...")
                logger.info(f"Index exists and matches (hash={current_hash[:8]}...)")
                logger.info("Loading existing index...")

                # Try LangChain API first
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    import torch  # noqa: F401 (kept for optional GPU flush)

                    logger.info(f"Using device for embeddings: {device}")
                    if progress_callback:
                        progress_callback("indexing", 30, f"Loading embeddings model on {device.upper()}...")

                    embedding = HuggingFaceEmbeddings(
                        model_name=resolved_model,
                        model_kwargs={"device": device},
                    )

                    try:
                        from langchain_chroma import Chroma
                        logger.debug("Using langchain_chroma (new API)")
                    except ImportError:
                        from langchain_community.vectorstores import Chroma
                        logger.debug("Using langchain_community.vectorstores (legacy)")

                    if progress_callback:
                        progress_callback("indexing", 50, "Loading vector database...")
                    vectordb = Chroma(
                        persist_directory=str(persist_dir),
                        embedding_function=embedding,
                        collection_name=collection_name,
                    )
                    logger.info("✓ Loaded existing index via LangChain")
                    if progress_callback:
                        progress_callback("indexing", 100, "✓ Index loaded successfully")
                    return vectordb

                except Exception as e:
                    logger.warning(f"Failed to load via LangChain: {e}")

                # Fallback: try chromadb directly
                try:
                    import chromadb

                    if progress_callback:
                        progress_callback("indexing", 50, "Loading vector database (chromadb)...")
                    client = chromadb.PersistentClient(path=str(persist_dir), settings=chromadb.Settings(anonymized_telemetry=False))
                    collection = client.get_collection(collection_name)
                    logger.info("✓ Loaded existing index via chromadb")
                    if progress_callback:
                        progress_callback("indexing", 100, "✓ Index loaded successfully")
                    return {"chroma_client": client, "collection": collection}

                except Exception as e:
                    logger.warning(f"Failed to load via chromadb: {e}")
                    logger.info("Will rebuild index...")

        # Load prepared documents
        if progress_callback:
            progress_callback("indexing", 20, "Loading prepared documents...")
        logger.info(f"Loading prepared documents: {prepared_path}")
        try:
            docs = self._load_prepared_jsonl(prepared_path)
            logger.info(f"Loaded {len(docs)} documents")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

        if not docs:
            raise ValueError("No documents to index")

        # Build embeddings using LangChain
        if progress_callback:
            progress_callback("indexing", 30, f"Building index with {len(docs)} documents...")
        logger.info(f"Building index with {len(docs)} documents...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_core.documents import Document

            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            import torch  # noqa: F401
            logger.info(f"Using device for embeddings: {device}")
            if progress_callback:
                progress_callback("indexing", 40, f"Loading embedding model on {device.upper()}...")

            embedding = HuggingFaceEmbeddings(
                model_name=resolved_model,
                model_kwargs={"device": device},
            )

            # Convert to LangChain Documents
            if progress_callback:
                progress_callback("indexing", 50, "Converting documents to LangChain format...")
            lc_docs = [
                Document(page_content=d["text"], metadata=d["metadata"]) for d in docs
            ]

            persist_dir.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback("indexing", 60, f"Creating vector index at {persist_dir}...")
            logger.info(f"Building Chroma index at {persist_dir}...")
            vectordb = Chroma.from_documents(
                documents=lc_docs,
                embedding=embedding,
                persist_directory=str(persist_dir),
                collection_name=collection_name,
            )

            # Try to persist (may not be needed in new API)
            try:
                vectordb.persist()
            except AttributeError:
                pass  # Not available in new API

            # Write metadata
            if progress_callback:
                progress_callback("indexing", 85, "Saving index metadata...")
            meta = {
                "hash": current_hash,
                "model": resolved_model,
                "n_docs": len(lc_docs),
                "collection_name": collection_name,
            }
            self._write_metadata(persist_dir, meta)

            logger.info(f"✓ Built index with {len(lc_docs)} documents")
            if progress_callback:
                progress_callback("indexing", 100, f"✓ Index built successfully ({len(lc_docs)} documents)")
            return vectordb

        except Exception as e:
            logger.warning(f"LangChain approach failed: {e}")
            logger.info("Falling back to sentence-transformers + chromadb...")
            if progress_callback:
                progress_callback("indexing", 50, "Falling back to sentence-transformers...")

            # Fallback: Use sentence-transformers directly
            try:
                from sentence_transformers import SentenceTransformer
                import chromadb
                from chromadb import errors as chromadb_errors
                import torch

                logger.info(f"Using device for embeddings: {device}")
                if progress_callback:
                    progress_callback("indexing", 60, f"Loading embedding model on {device.upper()}...")

                sbert = SentenceTransformer(resolved_model, device=device)

                texts = [d["text"] for d in docs]
                if progress_callback:
                    progress_callback("indexing", 70, f"Computing embeddings for {len(texts)} documents...")
                logger.info(f"Computing embeddings for {len(texts)} documents...")

                # Small batch size prevents OOM on low-VRAM GPUs; use even
                # smaller batches on cpu to keep memory pressure low.
                batch_size = 8 if device == "cuda" else 16
                embeddings = sbert.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )

                # Create chromadb index
                persist_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(persist_dir), settings=chromadb.Settings(anonymized_telemetry=False))

                # Create or get collection
                try:
                    collection = client.get_collection(collection_name)
                    logger.info(f"Updating existing collection: {collection_name}")
                except Exception:
                    try:
                        collection = client.create_collection(name=collection_name)
                        logger.info(f"Created new collection: {collection_name}")
                    except Exception as e:
                        # Race condition check
                        if (
                            chromadb_errors
                            and isinstance(e, chromadb_errors.InternalError)
                        ):
                            logger.warning(
                                "Collection creation race condition, retrieving..."
                            )
                        try:
                            collection = client.get_collection(collection_name)
                        except Exception as e2:
                            logger.error(f"Failed to get collection: {e2}")
                            raise

                # Upsert documents
                ids = [d["id"] for d in docs]
                metadatas = [d["metadata"] for d in docs]

                collection.upsert(
                    ids=ids,
                    metadatas=metadatas,
                    documents=texts,
                    embeddings=embeddings.tolist(),
                )

                # Write metadata
                meta = {
                    "hash": current_hash,
                    "model": resolved_model,
                    "n_docs": len(docs),
                    "collection_name": collection_name,
                }
                self._write_metadata(persist_dir, meta)

                logger.info(f"✓ Built index with {len(docs)} documents (chromadb)")
                return {"chroma_client": client, "collection": collection}

            except Exception as e:
                logger.error(f"All indexing approaches failed: {e}")
                raise RuntimeError(
                    f"Failed to build index: {e}. "
                    "Ensure following packages are installed: "
                    "langchain-huggingface, sentence-transformers, chromadb"
                ) from e
