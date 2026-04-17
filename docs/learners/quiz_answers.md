# Quiz Answers

## Module 0
1. venv isolates project dependencies.
2. `WEB_PORT` in `meetbot/config.py` / env.
3. `python -m meetbot.cli serve`.

## Module 1
1. transcribe → diarize → align → index.
2. DB `jobs` row + progress manager.
3. `results/<job_id>.json`.

## Module 2
1. `meetbot/adapters/transcribers/factory.py`.
2. To process large audio safely and stitch results.
3. It guides ASR language decoding.

## Module 3
1. `pyannote/speaker-diarization-3.1` by default.
2. diarization segments + transcription segments.
3. So text is labeled by speaker/time.

## Module 4
1. number of retrieved chunks.
2. `meetbot/services/query_service.py`.
3. under configured `VECTOR_DB_PATH` root.

## Module 5
1. quantized llama.cpp-compatible model format.
2. `LOCAL_LLM_GPU_LAYERS`.
3. switch to HF adapter mode.

## Module 6
1. using ORM objects after session close/lazy load.
2. reduce batch/load, retry on CPU.
3. `python -c "import torch,torchaudio;print(torch.__version__, torchaudio.__version__)"`.

## Module 7
1. app setup in `meetbot/web/main.py`.
2. `meetbot/services/query_service.py`.
3. run app + focused pytest.
