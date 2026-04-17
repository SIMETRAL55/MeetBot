# Tips for Small GPU (e.g., RTX 3050 4GB)

- Keep `LOCAL_LLM_GPU_LAYERS=0..8`.
- Prefer Q4 or Q3 GGUF models.
- Set `EMBEDDING_DEVICE=cpu` if indexing OOMs.
- Keep context small (`LOCAL_LLM_CONTEXT_SIZE=1024..2048`).
- Restart process between heavy runs to reduce fragmentation.
