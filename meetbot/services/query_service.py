"""RAG query service for retrieval-augmented generation.

LLM backend selection
---------------------
Pass ``llm_mode`` to :meth:`QueryService.query` to choose the generation backend:

* ``"local"``  – Use the locally-loaded GGUF model via llama.cpp.
* ``"hf"``     – Use the HuggingFace Inference API.

This mirrors the ``--use-local-llm`` flag in the CLI: only the final generation
step changes; retrieval, embedding and context-formatting are identical for both
modes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Callable
import threading

logger = logging.getLogger(__name__)

# Valid LLM mode literals
LLMMode = Literal["local", "hf"]

# Progress callback type: (stage: str, progress: float 0-100, message: str) -> None
ProgressCallback = Callable[[str, float, str], None]


# ---------------------------------------------------------------------------
# Module-level singleton for the HF adapter so the client is not recreated
# on every request.  The local LLM already has its own singleton via
# LocalLLMManager.get_instance().
# ---------------------------------------------------------------------------
_hf_adapter_cache: Optional[Any] = None


def _invalidate_chroma_cache(job_id: str) -> None:
    """Clear chromadb's process-level SharedSystemClient registry.

    chromadb 0.4+ caches ``PersistentClient`` instances in
    ``SharedSystemClient._identifier_to_system`` keyed by the persist
    directory path.  When the underlying SQLite files are deleted by
    ``shutil.rmtree`` the cached System still holds open file descriptors
    to the now-deleted inodes.  Two consequences follow:

    * **Stale query results** — the next ``_load_vectorstore`` call hits the
      cache and reads from the old inode (still accessible via FD on Linux),
      not the newly written files.
    * **SQLITE_READONLY_DBMOVED (code 1032)** — on the second reindex,
      ``Chroma.from_documents`` receives the cached System whose ``SqliteDB``
      references a path that no longer exists, so every write fails.

    Calling this function **before** ``shutil.rmtree`` releases the file
    handles so the filesystem tree can be safely deleted, and calling it
    **after** building the new index ensures the next query opens a fresh
    client that reads the newly-written SQLite.
    """
    import gc

    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
        logger.debug(
            "Cleared chromadb SharedSystemClient cache (job=%s)",
            job_id[:8] if job_id else "?",
        )
    except Exception as exc:
        logger.debug("chromadb cache clear skipped: %s", exc)

    # Force reference-counting GC so System objects (and their SQLite FDs)
    # are released before the caller proceeds with rmtree / index build.
    gc.collect()


def _get_hf_adapter(model: str) -> Any:
    """Return a cached HFAPILLMAdapter, creating it if needed.

    The adapter is rebuilt when either the model ID or the configured
    HF_PROVIDER changes.  Provider auto-resolution (HF_PROVIDER=auto)
    queries the HF model-card API once and caches the result.
    """
    global _hf_adapter_cache
    from ..adapters.llm import HFAPILLMAdapter
    from ..config import settings

    wanted_provider = settings.HF_PROVIDER  # "auto" or a concrete name

    # Rebuild when model or provider setting changes
    if (
        _hf_adapter_cache is None
        or getattr(_hf_adapter_cache, "model", None) != model
        or getattr(_hf_adapter_cache, "_wanted_provider", None) != wanted_provider
    ):
        adapter = HFAPILLMAdapter(
            token=settings.get_hf_token(),
            model=model,
            provider=wanted_provider,
        )
        # Stash the *setting* so we can detect config changes
        adapter._wanted_provider = wanted_provider  # type: ignore[attr-defined]
        _hf_adapter_cache = adapter
        logger.info(
            "HFAPILLMAdapter created: model=%s, provider=%s (config=%s)",
            model,
            adapter.provider,
            wanted_provider,
        )
    return _hf_adapter_cache


class QueryService:
    """
    RAG query service for retrieval-augmented generation.

    Retrieves relevant documents from vector store and generates
    answers using local or remote LLM.
    """

    def __init__(self):
        """Initialize query service."""
        pass

    @staticmethod
    def _load_vectorstore(db_dir: str, embedding_model: str) -> Any:
        """
        Load Chroma vector store with embeddings.

        Tries LangChain API first, falls back to chromadb directly.

        Args:
            db_dir: Path to Chroma database directory
            embedding_model: Embedding model name/path

        Returns:
            Vectorstore object (Chroma or chromadb collection)

        Raises:
            RuntimeError: If unable to load vectorstore
        """
        from pathlib import Path

        db_path = Path(db_dir)
        if not db_path.exists():
            raise FileNotFoundError(f"Vector database not found: {db_dir}")

        collection_name = db_path.name

        # Try LangChain API
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            logger.debug("Loading vectorstore via LangChain...")
            embedding = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},
            )
            vectordb = Chroma(
                persist_directory=db_dir,
                embedding_function=embedding,
                collection_name=collection_name,
            )
            logger.info(f"✓ Loaded vectorstore from {db_dir}")
            return vectordb

        except Exception as e:
            logger.warning(f"LangChain loading failed: {e}")

        # Fallback: chromadb direct API
        try:
            import chromadb

            logger.debug("Falling back to chromadb direct API...")
            client = chromadb.PersistentClient(path=db_dir)
            collection = client.get_collection(collection_name)
            logger.info(f"✓ Loaded vectorstore via chromadb from {db_dir}")
            return {"chroma_client": client, "collection": collection}

        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            raise RuntimeError(f"Could not load vectorstore from {db_dir}: {e}") from e

    @staticmethod
    def _create_retriever(vectorstore: Any, k: int = 4) -> Any:
        """
        Create retriever from vectorstore.

        Supports both LangChain and chromadb APIs.

        Args:
            vectorstore: Vectorstore object
            k: Number of documents to retrieve

        Returns:
            Retriever callable
        """
        # LangChain vectorstore
        if hasattr(vectorstore, "as_retriever"):
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": k})

                class LangChainRetriever:
                    def __init__(self, ret):
                        self.retriever = ret

                    def retrieve(self, query: str) -> List[Any]:
                        # Try new API first
                        if hasattr(self.retriever, "invoke"):
                            return self.retriever.invoke(query)
                        # Fall back to old API
                        elif hasattr(self.retriever, "get_relevant_documents"):
                            return self.retriever.get_relevant_documents(query)
                        else:
                            raise AttributeError(
                                f"Retriever {type(self.retriever)} has no invoke or get_relevant_documents"
                            )

                logger.debug("Using LangChain retriever")
                return LangChainRetriever(retriever)

            except Exception as e:
                logger.warning(f"LangChain retriever failed: {e}")

        # Chromadb collection
        if isinstance(vectorstore, dict) and "collection" in vectorstore:
            collection = vectorstore["collection"]

            class ChromaDbRetriever:
                def __init__(self, coll, k):
                    self.collection = coll
                    self.k = k

                def retrieve(self, query: str) -> List[Any]:
                    # Query chromadb collection
                    results = self.collection.query(query_texts=[query], n_results=self.k)

                    # Convert to document-like objects
                    documents = []
                    if results and "documents" in results:
                        for i, text in enumerate(results["documents"][0]):
                            metadata = (
                                results["metadatas"][0][i]
                                if i < len(results["metadatas"][0])
                                else {}
                            )
                            doc = type("Document", (), {
                                "page_content": text,
                                "metadata": metadata,
                            })()
                            documents.append(doc)

                    return documents

            logger.debug("Using chromadb retriever")
            return ChromaDbRetriever(collection, k)

        # Fallback with similarity_search
        try:
            class SimpleRetriever:
                def __init__(self, vs, k):
                    self.vectorstore = vs
                    self.k = k

                def retrieve(self, query: str) -> List[Any]:
                    return self.vectorstore.similarity_search(query, k=self.k)

            logger.debug("Using simple similarity_search retriever")
            return SimpleRetriever(vectorstore, k)

        except Exception as e:
            logger.error(f"Could not create retriever: {e}")
            raise RuntimeError(f"Failed to create retriever: {e}") from e

    @staticmethod
    def _get_local_adapter() -> Any:
        """
        Return a cached LocalLLMAdapter instance.

        Uses the LocalLLMManager singleton so the model is never reloaded
        between requests.

        Returns:
            LocalLLMAdapter instance

        Raises:
            RuntimeError: If llama_cpp is not installed or model path is invalid
        """
        from ..adapters.llm import LocalLLMAdapter
        from ..config import settings

        logger.info("Using Local LLM")
        return LocalLLMAdapter(
            model_path=settings.LOCAL_LLM_MODEL_PATH,
            n_gpu_layers=settings.LOCAL_LLM_GPU_LAYERS,
        )

    @staticmethod
    def _get_llm_model(use_local: Optional[bool] = None) -> Optional[Any]:
        """
        Backward-compatible helper: return a local LLM adapter when enabled.

        Args:
            use_local: Override use_local_llm setting

        Returns:
            LLM instance if available, None otherwise
        """
        from ..adapters.llm import get_local_llm
        from ..config import settings

        enabled = use_local if use_local is not None else settings.USE_LOCAL_LLM
        if not enabled:
            logger.debug("Local LLM disabled")
            return None

        try:
            if get_local_llm is None:
                logger.debug("Local LLM module not available")
                return None

            logger.info("Loading local LLM...")
            llm = get_local_llm(
                model_path=settings.LOCAL_LLM_MODEL_PATH,
                n_gpu_layers=settings.LOCAL_LLM_GPU_LAYERS,
            )
            return llm

        except Exception as e:
            logger.warning(f"Failed to load local LLM: {e}")
            return None

    @staticmethod
    def _call_local_llm(llm: Any, system_msg: str, user_content: str) -> str:
        """
        Call local LLM with instruction format.

        Args:
            llm: Local LLM instance
            system_msg: System instructions
            user_content: User query with context

        Returns:
            Generated response text
        """
        # Format for RakutenAI-7B-Instruct
        prompt = f"[INST] {system_msg}\n\n{user_content} [/INST]"

        logger.info(f"Calling local LLM (prompt length={len(prompt)} chars)...")
        response = llm.generate(prompt)

        logger.info(f"Received response ({len(response)} chars)")
        return response

    @staticmethod
    def _call_hf_llm(hf_model: str, messages: List[Dict[str, str]]) -> str:
        """
        Call HuggingFace API LLM using the HFAPILLMAdapter.

        The adapter handles token validation, provider routing, and retries.
        A module-level singleton prevents recreating the client on every call.

        Args:
            hf_model: HuggingFace model ID
            messages: List of chat messages (role / content dicts)

        Returns:
            Generated response text

        Raises:
            RuntimeError: If API key is missing, rate-limited, or call fails
        """
        import os

        logger.info("Using HuggingFace Inference")

        # Validate token exists before incurring network latency
        token = (
            os.environ.get("HF_API_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HF_HUB_TOKEN")
        )
        if not token:
            raise RuntimeError(
                "HF_API_TOKEN is not set. "
                "Export HF_API_TOKEN=hf_... before starting the server."
            )

        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError("huggingface_hub not installed")

        provider = os.environ.get("HF_PROVIDER", "fireworks-ai")
        logger.info(f"Calling HF model {hf_model} (provider={provider})...")

        client = InferenceClient(provider=provider, api_key=token)
        try:
            completion = client.chat.completions.create(
                model=hf_model,
                messages=messages,
            )
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate limit" in err_str.lower():
                raise RuntimeError(
                    f"HuggingFace rate limit exceeded (429). "
                    f"Wait a moment and retry. Details: {err_str}"
                ) from exc
            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                raise RuntimeError(
                    f"HuggingFace API request timed out. Details: {err_str}"
                ) from exc
            raise RuntimeError(f"HuggingFace API call failed: {err_str}") from exc

        # Extract message content
        try:
            first_choice = completion.choices[0]
        except Exception:
            if hasattr(completion, "get"):
                choices = completion.get("choices", [])
                if not choices:
                    raise RuntimeError("No choices in HF response")
                first_choice = choices[0]
            else:
                raise RuntimeError(f"Unexpected HF response: {completion}")

        message = (
            first_choice.message
            if hasattr(first_choice, "message")
            else first_choice.get("message", first_choice)
        )

        if isinstance(message, dict):
            content = message.get("content") or message.get("text") or ""
        elif hasattr(message, "content"):
            content = message.content
        else:
            content = str(message)

        content_text = str(content).strip()
        logger.info(f"Received response ({len(content_text)} chars)")
        return content_text

    def query(
        self,
        question: str,
        db_dir: str,
        embedding_model: str = "./models/sarashina-embedding-v1-1b",
        hf_model: str = "deepseek-ai/DeepSeek-V3.1",
        k: int = 4,
        use_local_llm: Optional[bool] = None,
        llm_mode: Optional[LLMMode] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Answer question using RAG pipeline.

        Retrieves relevant documents and generates answer using LLM.

        LLM backend selection (both ``llm_mode`` and ``use_local_llm`` are
        supported for backward-compatibility):

        * ``llm_mode="local"``  → LocalLLMAdapter (llama.cpp GGUF)
        * ``llm_mode="hf"``     → HuggingFace Inference API
        * If ``llm_mode`` is not supplied then ``use_local_llm`` and / or
          ``settings.USE_LOCAL_LLM`` is used as before.

        Args:
            question: User question
            db_dir: Path to vector database
            embedding_model: Embedding model for search
            hf_model: HuggingFace model for generation (if local LLM disabled)
            k: Number of documents to retrieve
            use_local_llm: Legacy override; ignored when llm_mode is set
            llm_mode: Explicit LLM backend: "local" | "hf"  (takes priority)
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with keys:
                - answer: Generated answer text
                - retrieved: List of (text, metadata) tuples
                - sources: Formatted source information
                - llm_backend: Which backend was used ("local" | "hf")

        Raises:
            RuntimeError: If query execution fails
        """
        # ── Resolve effective LLM mode ──────────────────────────────────────
        # Priority: explicit llm_mode > use_local_llm param > settings default
        if llm_mode is not None:
            effective_mode: LLMMode = llm_mode
        elif use_local_llm is not None:
            effective_mode = "local" if use_local_llm else "hf"
        else:
            from ..config import settings as _s
            effective_mode = "local" if _s.USE_LOCAL_LLM else "hf"

        if progress_callback:
            progress_callback("query", 5, "Initializing RAG query...")
        logger.info(f"Starting RAG query: {question[:100]}... (llm_mode={effective_mode})")

        # Load vectorstore
        if progress_callback:
            progress_callback("query", 15, "Loading vector database...")
        logger.info("Loading vector database...")
        vectorstore = self._load_vectorstore(db_dir, embedding_model)

        # Create retriever
        if progress_callback:
            progress_callback("query", 25, f"Creating retriever (k={k})...")
        logger.info(f"Creating retriever (k={k})...")
        retriever = self._create_retriever(vectorstore, k=k)

        # Retrieve documents
        if progress_callback:
            progress_callback("query", 35, f"Retrieving top {k} relevant documents...")
        logger.info(f"Retrieving top {k} documents...")
        docs = retriever.retrieve(question)

        # Process retrieved documents
        if progress_callback:
            progress_callback("query", 45, "Processing retrieved documents...")
        context: List[Tuple[str, Dict[str, Any]]] = []
        for doc in docs:
            text = getattr(doc, "page_content", doc)
            metadata = getattr(doc, "metadata", {})
            context.append((text, metadata))

        logger.info(f"Retrieved {len(context)} documents")

        # Format context for LLM
        if progress_callback:
            progress_callback("query", 55, "Formatting context for LLM...")
        context_text = "\n\n---\n\n".join(
            f"[{m.get('audio_file')} {m.get('speaker')} "
            f"{m.get('start', '?')}-{m.get('end', '?')}] {txt}"
            for txt, m in context
        )

        # Build messages for LLM
        system_msg = (
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the answer is not in the context, say so."
        )
        user_content = f"{context_text}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        logger.info(f"Prepared context (length={len(user_content)} chars)")

        # ── Strategy dispatch: call the selected LLM backend ───────────────
        if progress_callback:
            progress_callback("query", 65, "Generating answer using LLM...")

        if effective_mode == "local":
            if progress_callback:
                progress_callback("query", 70, "Calling Local LLM (llama.cpp)...")
            logger.info("Using Local LLM")
            try:
                llm = self._get_local_adapter()
                answer_text = self._call_local_llm(llm, system_msg, user_content)
            except RuntimeError as e:
                err_str = str(e)
                # Wrap OOM into a user-friendly message
                if "out of memory" in err_str.lower() or "cuda" in err_str.lower():
                    friendly = (
                        "Local LLM failed: GPU out of memory. "
                        "Try switching to HuggingFace backend or restarting the server "
                        "to free VRAM."
                    )
                else:
                    friendly = f"Local LLM failed: {err_str}"
                logger.error("Local LLM error: %s", err_str)
                if progress_callback:
                    progress_callback("query", 0, f"✗ {friendly}")
                raise RuntimeError(friendly) from e
            except Exception as e:
                logger.error("Local LLM unexpected error: %s", e)
                if progress_callback:
                    progress_callback("query", 0, f"✗ Local LLM failed: {e}")
                raise RuntimeError(f"Local LLM failed: {e}") from e

        else:  # effective_mode == "hf"
            if progress_callback:
                progress_callback("query", 70, "Calling HuggingFace Inference API...")
            logger.info("Using HuggingFace Inference")
            try:
                answer_text = self._call_hf_llm(hf_model, messages)
            except RuntimeError as e:
                err_str = str(e)
                if progress_callback:
                    progress_callback("query", 0, f"✗ HF Inference failed: {err_str}")
                raise
            except Exception as e:
                logger.error("HF API unexpected error: %s", e)
                if progress_callback:
                    progress_callback("query", 0, f"✗ HF API failed: {e}")
                raise RuntimeError(f"HuggingFace Inference failed: {e}") from e

        # Format sources
        if progress_callback:
            progress_callback("query", 85, "Formatting results...")
        sources = [
            {
                "text": text[:200],  # First 200 chars
                "audio_file": m.get("audio_file"),
                "speaker": m.get("speaker"),
                "start": m.get("start"),
                "end": m.get("end"),
            }
            for text, m in context
        ]

        result = {
            "answer": answer_text,
            "retrieved": context,
            "sources": sources,
            "n_sources": len(sources),
            "llm_backend": effective_mode,
        }

        if progress_callback:
            progress_callback("query", 100, f"✓ Query complete ({len(sources)} sources)")
        logger.info("✓ Query completed successfully")
        return result

    def query_stream(
        self,
        question: str,
        db_dir: str,
        embedding_model: str = "./models/sarashina-embedding-v1-1b",
        hf_model: str = "deepseek-ai/DeepSeek-V3.1",
        k: int = 4,
        llm_mode: LLMMode = "local",
        abort_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        RAG query with streaming LLM generation.

        This is a **synchronous generator** — run it in a background thread
        (e.g. via ``asyncio.to_thread``) to avoid blocking the asyncio event
        loop.

        Emits a sequence of typed event dicts:

        1. ``{"type": "sources", "data": [...]}``
           Emitted immediately after retrieval, before generation begins.
           Each source is a dict with speaker, start, end, text, audio_file.

        2. ``{"type": "token", "data": "<str>"}``
           One per generated token/sub-word.  Concatenate all token data to
           reconstruct the full answer.

        3. ``{"type": "done", "llm_backend": "local"|"hf", "full_answer": "..."}``
           Emitted when generation is complete.

        On error, emits ``{"type": "error", "data": "<message>"}`` and stops.

        Args:
            question: User question string.
            db_dir: Path to ChromaDB vector store.
            embedding_model: Embedding model for retrieval.
            hf_model: HuggingFace model ID (used only when llm_mode="hf").
            k: Number of documents to retrieve.
            llm_mode: "local" | "hf"

        Yields:
            dict: Typed event dicts as described above.
        """
        # ── Step 1: Retrieval ──────────────────────────────────────────────
        try:
            logger.info("query_stream: loading vectorstore from %s", db_dir)
            vectorstore = self._load_vectorstore(db_dir, embedding_model)
            retriever   = self._create_retriever(vectorstore, k=k)
            docs        = retriever.retrieve(question)
        except Exception as exc:
            logger.error("query_stream retrieval failed: %s", exc)
            yield {"type": "error", "data": f"Retrieval failed: {exc}"}
            return

        # ── Step 2: Process documents & build context ──────────────────────
        context: List[Tuple[str, Dict[str, Any]]] = []
        for doc in docs:
            text     = getattr(doc, "page_content", doc)
            metadata = getattr(doc, "metadata", {})
            context.append((text, metadata))

        sources = [
            {
                "text":       text[:200],
                "audio_file": m.get("audio_file"),
                "speaker":    m.get("speaker"),
                "start":      m.get("start"),
                "end":        m.get("end"),
            }
            for text, m in context
        ]

        # Emit sources immediately — client shows them before generation starts
        yield {"type": "sources", "data": sources}
        logger.info("query_stream: emitted %d sources", len(sources))

        # ── Step 3: Build prompt ───────────────────────────────────────────
        context_text = "\n\n---\n\n".join(
            f"[{m.get('audio_file')} {m.get('speaker')} "
            f"{m.get('start', '?')}-{m.get('end', '?')}] {txt}"
            for txt, m in context
        )
        system_msg  = (
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the answer is not in the context, say so."
        )
        user_content = f"{context_text}\n\nQuestion: {question}"
        prompt       = f"[INST] {system_msg}\n\n{user_content} [/INST]"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_content},
        ]

        # ── Step 4: Stream generation ──────────────────────────────────────
        from ..config import settings as _s

        full_answer = ""
        stopped = False
        try:
            if llm_mode == "local":
                logger.info("query_stream: using Local LLM")
                adapter = self._get_local_adapter()
                token_iter = adapter.generate_stream(
                    prompt=prompt,
                    max_tokens=_s.LOCAL_LLM_MAX_TOKENS,
                    temperature=_s.LOCAL_LLM_TEMPERATURE,
                )
            else:
                logger.info("query_stream: using HuggingFace Inference (model=%s)", hf_model)
                hf_adapter = _get_hf_adapter(hf_model)
                # Build flat prompt for text_generation endpoint
                hf_prompt = f"[INST] {system_msg}\n\n{user_content} [/INST]"
                token_iter = hf_adapter.generate_stream(prompt=hf_prompt)

            for token in token_iter:
                # Cooperative abort check between tokens
                if abort_event is not None and abort_event.is_set():
                    stopped = True
                    break
                full_answer += token
                yield {"type": "token", "data": token}

        except RuntimeError as exc:
            err_str = str(exc)
            logger.error("query_stream generation error: %s", err_str)
            yield {"type": "error", "data": err_str}
            return
        except Exception as exc:
            logger.error("query_stream unexpected error: %s", exc)
            yield {"type": "error", "data": f"Generation failed: {exc}"}
            return

        yield {"type": "done", "llm_backend": llm_mode, "full_answer": full_answer, "stopped": stopped}
