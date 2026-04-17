# Debug Recipes

## 1) `nvidia-smi` driver errors
- **Reproduce**: run `nvidia-smi` and get failure/no devices.
- **Root cause**: host driver mismatch or container missing `--gpus all`.
- **Fix**:
```bash
nvidia-smi
docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
- Check host driver installation and NVIDIA container toolkit.

## 2) torch/torchaudio compatibility errors
- **Check versions**:
```bash
python -c "import torch,torchaudio;print(torch.__version__, torchaudio.__version__)"
pip show torch torchaudio
```
- **Fix**: reinstall matching wheels from same index URL (cpu/cu121 etc.).

## 3) DetachedInstanceError
- **Reproduce**: access lazy ORM attribute after `db.close()`.
- **Fix**: extract primitive fields before close or use eager loading.

## 4) Chroma readonly DB
- **Symptom**: sqlite readonly/lock errors on reindex.
- **Fix pattern**: build into temp dir then atomic replace.
```bash
# pseudo steps
rm -rf db/job123_tmp
# build tmp index
mv db/job123 db/job123_old && mv db/job123_tmp db/job123
```

## 5) CUDA out of memory
- **Reproduce**: run pipeline on long file with GPU embedding.
- **Fix**:
```bash
export EMBEDDING_DEVICE=cpu
export LOCAL_LLM_GPU_LAYERS=0
```
- worker already retries CPU on indexing OOM.

## 6) NiceGUI client/request mismatch
- **Symptom**: callback context errors when using UI objects outside request/client scope.
- **Fix**: keep UI updates in proper callback context and avoid cross-client global state.

## 7) HF gated model 403
- **Reproduce**: diarization call returns 401/403.
- **Fix**:
```bash
export HF_API_TOKEN=hf_xxx
```
Accept model license pages on Hugging Face (pyannote model card).

## 8) Upload event attribute changes (NiceGUI)
- **Symptom**: `UploadEventArguments` missing expected attr.
- **Fix**: inspect event object fields in current NiceGUI version and adapt handler.
```python
# debug snippet
print(dir(e), e.__dict__)
```
