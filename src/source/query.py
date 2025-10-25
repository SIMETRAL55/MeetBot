# query.py
from pathlib import Path
import os
import argparse
import logging
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("query")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_vectorstore(db_dir: str, embed_model: str):
    """
    Load the vector store with embeddings.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if token:
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)

    # Use local embeddings
    emb = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cpu"})
    # emb = HuggingFaceEmbeddings(model_name=embed_model)
    vectordb = Chroma(
        persist_directory=db_dir,
        embedding_function=emb,
        collection_name=Path(db_dir).name
    )
    logger.info("Loaded Chroma DB from %s", db_dir)
    return vectordb


def _create_retriever(vectordb, k: int = 4):
    try:
        return vectordb.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.warning("as_retriever failed, fallback to similarity_search: %s", e)
        class SimpleRetriever:
            def __init__(self, db, k):
                self.db = db
                self.k = k
            def get_relevant_documents(self, q: str):
                return self.db.similarity_search(q, k=self.k)
        return SimpleRetriever(vectordb, k=k)

def _call_hf_chat_model(hf_model: str, messages: List[Dict[str, str]]) -> str:
    try:
        from huggingface_hub import InferenceClient
    except Exception as e:
        logger.exception("Please install huggingface_hub (pip install huggingface-hub): %s", e)
        raise

    token = (
        os.environ.get("HF_API_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError("No HF token found. Set HF_API_TOKEN or HUGGINGFACEHUB_API_TOKEN/HF_HUB_TOKEN.")

    provider = os.environ.get("HF_PROVIDER", "fireworks-ai")

    client = InferenceClient(provider=provider, api_key=token)

    logger.info("Sending chat completion to HF model %s (provider=%s)", hf_model, provider)
    # call chat completions
    completion = client.chat.completions.create(
        model=hf_model,
        messages=messages,
    )

    try:
        first_choice = completion.choices[0]
    except Exception:
        # fallback to dict-like
        if hasattr(completion, "get"):
            choices = completion.get("choices", [])
            if not choices:
                raise RuntimeError("No choices returned from HF API")
            first_choice = choices[0]
        else:
            raise RuntimeError("Unexpected response from HF API: %r" % completion)

    # obtain message
    message = None
    if hasattr(first_choice, "message"):
        message = first_choice.message
    elif isinstance(first_choice, dict):
        message = first_choice.get("message")
    else:
        message = first_choice

    # message might be dict with "content" or similar shapes; normalize to string
    content_text = ""
    if isinstance(message, dict):
        # message content might be string or dict or list; handle common variants
        content = message.get("content") or message.get("text") or message.get("parts") or message
        # if content is dict like {"type": "text", "text": "..."} or {"content": "..."}
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, dict):
            # try nested keys
            for k in ("content", "text", "value", "message"):
                if k in content and isinstance(content[k], str):
                    content_text = content[k]
                    break
            if not content_text:
                # as last resort, stringify
                content_text = str(content)
        elif isinstance(content, (list, tuple)):
            # join parts
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    # common format: {"type": "output_text", "text": "..."}
                    ptxt = part.get("text") or part.get("content")
                    if ptxt:
                        parts.append(ptxt)
                    else:
                        parts.append(str(part))
                else:
                    parts.append(str(part))
            content_text = "".join(parts)
        else:
            content_text = str(content)
    else:
        # message is not a dict; fallback to str()
        content_text = str(message)

    # Trim whitespace
    content_text = content_text.strip()

    logger.info("Received %d chars from HF model", len(content_text))
    return content_text


def answer_query(
    db_dir: str,
    question: str,
    embed_model: str,
    hf_model: str,
    k: int = 4
) -> Dict[str, Any]:
    vectordb = _load_vectorstore(db_dir, embed_model)
    retriever = _create_retriever(vectordb, k=k)
    docs = retriever.get_relevant_documents(question)
    # print(docs)

    context: List[Tuple[str, Dict[str, Any]]] = []
    for d in docs:
        text = getattr(d, "page_content", d)
        meta = getattr(d, "metadata", {})
        context.append((text, meta))

    # build the prompt for chat-style models using messages list
    ctx_text = "\n\n---\n\n".join(
        f"[{m.get('audio_file')} {m.get('speaker')} {m.get('start')}-{m.get('end')}] {txt}"
        for txt, m in context
    )

    # Prepare messages: system + user
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Answer the user's question using only the provided context. If the answer is not contained in the context, say you don't know."
    }
    user_content = f"{ctx_text}\n\nQuestion: {question}"
    user_msg = {"role": "user", "content": user_content}

    messages = [system_msg, user_msg]

    logger.info("Messages prepared for HF chat (user message length=%d chars)", len(user_content))

    # call HF chat API (DeepSeek style)
    try:
        answer_text = _call_hf_chat_model(hf_model, messages)
    except Exception as e:
        logger.exception("Chat call failed: %s", e)
        raise

    return {"answer": answer_text, "retrieved": context}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-root", default="db", help="root folder for DBs")
    parser.add_argument("--audio", required=True, help="audio basename (folder in db/)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--embed-model", default="sbintuitions/sarashina-embedding-v1-1b")
    parser.add_argument("--hf-model", default="deepseek-ai/DeepSeek-V3.1", help="HF inference model repo, e.g. deepseek-ai/DeepSeek-V3.1")
    args = parser.parse_args()

    db_dir = Path(args.db_root) / args.audio
    res = answer_query(str(db_dir), args.question, args.embed_model, args.hf_model, k=args.k)

    print("=== ANSWER ===")
    print(res["answer"])
    print("\n=== SOURCES ===")
    for txt, meta in res["retrieved"]:
        print(f"- {meta.get('audio_file')} {meta.get('speaker')} [{meta.get('start')}-{meta.get('end')}] -> {txt[:200]}")


if __name__ == "__main__":
    main()
