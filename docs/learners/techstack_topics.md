# Tech Stack Topics to Study

Format: **Topic** — required/optional, what it is, why MeetBot uses it, resources.

1. **Python basics (functions, classes, async, threading)** — **required**  
Used across workers, web callbacks, and adapters.  
Resources: https://docs.python.org/3/tutorial/ ; https://realpython.com/python-concurrency/ ; https://docs.python.org/3/library/threading.html

2. **Virtual environments + pip + requirements** — **required**  
Needed to isolate torch/torchaudio and reproducible installs.  
Resources: https://docs.python.org/3/library/venv.html ; https://pip.pypa.io/en/stable/user_guide/ ; https://packaging.python.org/

3. **FastAPI + NiceGUI page registration** — **required**  
MeetBot’s UI is NiceGUI on top of FastAPI routes/ws.  
Resources: https://fastapi.tiangolo.com/ ; https://nicegui.io/documentation ; https://github.com/zauberzeug/nicegui

4. **WebSockets vs polling** — **required**  
Progress/chat use WS for low-latency updates instead of repeated polling.  
Resources: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API ; https://fastapi.tiangolo.com/advanced/websockets/

5. **SQLAlchemy ORM + session lifecycle (detached instances)** — **required**  
App closes sessions often between UI callbacks; detached objects can fail later.  
Resources: https://docs.sqlalchemy.org/en/20/orm/session_basics.html ; https://docs.sqlalchemy.org/en/20/errors.html ; https://docs.sqlalchemy.org/en/20/orm/loading_relationships.html

6. **SQLite quirks (permissions, WAL)** — **required**  
Single-file DB + WAL side files appear in this project.  
Resources: https://sqlite.org/wal.html ; https://docs.sqlalchemy.org/en/20/dialects/sqlite.html

7. **Background worker patterns (queue + threads)** — **required**  
MeetBot pipeline is long-running and must not block HTTP/UI request path.  
Resources: https://docs.python.org/3/library/queue.html ; https://superfastpython.com/thread-queue/

8. **Audio processing fundamentals (ffmpeg, sample rates, mono/stereo)** — **required**  
Preprocessing audio shape and chunking strongly impacts ASR/diarization quality.  
Resources: https://ffmpeg.org/ffmpeg.html ; https://pydub.com/

9. **Whisper local + HF ASR** — **required**  
Transcription backend is switchable; performance/accuracy tradeoffs matter.  
Resources: https://github.com/openai/whisper ; https://huggingface.co/docs/transformers/model_doc/whisper ; https://huggingface.co/docs/inference-providers

10. **Pyannote diarization basics** — **required**  
Speaker segmentation is core to “who said what.”  
Resources: https://github.com/pyannote/pyannote-audio ; https://huggingface.co/pyannote/speaker-diarization-3.1

11. **Sentence transformers / embeddings** — **required**  
Converts transcript chunks to vectors for semantic retrieval.  
Resources: https://www.sbert.net/ ; https://huggingface.co/sentence-transformers

12. **Vector DBs (Chroma) + LangChain integration** — **required**  
Persistent retrieval index for RAG queries.  
Resources: https://docs.trychroma.com/ ; https://python.langchain.com/docs/integrations/vectorstores/chroma/

13. **Quantization (GGUF Q4/Q3), llama.cpp, ctransformers** — **required**  
Enables local LLM inference on smaller GPUs/CPUs.  
Resources: https://github.com/ggml-org/llama.cpp ; https://huggingface.co/docs/hub/gguf ; https://github.com/marella/ctransformers

14. **bitsandbytes, torch, torchaudio compatibility** — **required**  
Most common setup breakage in ML Python stacks.  
Resources: https://pytorch.org/get-started/locally/ ; https://github.com/TimDettmers/bitsandbytes ; https://pytorch.org/audio/stable/

15. **Hugging Face tokens, gated models, providers (and FAL/provider errors)** — **required**  
Diarization and some LLM models require accepted terms + valid scoped tokens.  
Resources: https://huggingface.co/docs/hub/security-tokens ; https://huggingface.co/docs/inference-providers/index

16. **Docker basics + GPU containers** — **optional**  
Useful for reproducible deployment and driver/runtime isolation.  
Resources: https://docs.docker.com/get-started/ ; https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/

17. **Debugging ML web apps (OOM, API drift, session bugs)** — **required**  
MeetBot combines web + ML + DB; failures are usually integration issues.  
Resources: https://fastapi.tiangolo.com/tutorial/debugging/ ; https://docs.python.org/3/howto/logging.html

18. **Security/privacy (local vs cloud model inference)** — **required**  
Meeting data sensitivity drives architecture decisions.  
Resources: https://owasp.org/www-project-top-ten/ ; https://huggingface.co/docs/hub/security ; https://fastapi.tiangolo.com/advanced/security/
