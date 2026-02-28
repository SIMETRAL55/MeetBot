"""RAG query service for retrieval-augmented generation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
    def _get_llm_model(use_local: Optional[bool] = None) -> Optional[Any]:
        """
        Get LLM instance if local LLM enabled.

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
        Call HuggingFace API LLM with chat messages.

        Args:
            hf_model: HuggingFace model ID
            messages: List of chat messages

        Returns:
            Generated response text

        Raises:
            RuntimeError: If API call fails
        """
        import os

        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError("huggingface_hub not installed")

        # Get API token
        token = (
            os.environ.get("HF_API_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HF_HUB_TOKEN")
        )

        if not token:
            raise RuntimeError("HF_API_TOKEN not set in environment")

        provider = os.environ.get("HF_PROVIDER", "fireworks-ai")
        logger.info(f"Calling HF model {hf_model} (provider={provider})...")

        client = InferenceClient(provider=provider, api_key=token)
        completion = client.chat.completions.create(
            model=hf_model,
            messages=messages,
        )

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

        # Get message content
        message = (
            first_choice.message
            if hasattr(first_choice, "message")
            else first_choice.get("message", first_choice)
        )

        # Extract text from message
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
    ) -> Dict[str, Any]:
        """
        Answer question using RAG pipeline.

        Retrieves relevant documents and generates answer using LLM.

        Args:
            question: User question
            db_dir: Path to vector database
            embedding_model: Embedding model for search
            hf_model: HuggingFace model for generation (if local LLM disabled)
            k: Number of documents to retrieve
            use_local_llm: Override use_local_llm setting

        Returns:
            Dict with keys:
                - answer: Generated answer text
                - retrieved: List of (text, metadata) tuples
                - sources: Formatted source information

        Raises:
            RuntimeError: If query execution fails
        """
        logger.info(f"Starting RAG query: {question[:100]}...")

        # Load vectorstore
        logger.info("Loading vector database...")
        vectorstore = self._load_vectorstore(db_dir, embedding_model)

        # Create retriever
        logger.info(f"Creating retriever (k={k})...")
        retriever = self._create_retriever(vectorstore, k=k)

        # Retrieve documents
        logger.info(f"Retrieving top {k} documents...")
        docs = retriever.retrieve(question)

        # Process retrieved documents
        context: List[Tuple[str, Dict[str, Any]]] = []
        for doc in docs:
            text = getattr(doc, "page_content", doc)
            metadata = getattr(doc, "metadata", {})
            context.append((text, metadata))

        logger.info(f"Retrieved {len(context)} documents")

        # Format context for LLM
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

        # Call LLM
        llm = self._get_llm_model(use_local_llm)
        if llm:
            logger.info("Using local LLM for generation")
            try:
                answer_text = self._call_local_llm(llm, system_msg, user_content)
            except Exception as e:
                logger.error(f"Local LLM failed: {e}")
                raise
        else:
            logger.info("Using HuggingFace API for generation")
            try:
                answer_text = self._call_hf_llm(hf_model, messages)
            except Exception as e:
                logger.error(f"HF API failed: {e}")
                raise

        # Format sources
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
        }

        logger.info("✓ Query completed successfully")
        return result
