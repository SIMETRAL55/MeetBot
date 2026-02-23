import logging
import os
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import InferenceClient

from meetbot.config.settings import settings

logger = logging.getLogger(__name__)


class HFInferenceClient:
    """Adapter for Hugging Face inference + local pyannote."""

    def __init__(self, token: Optional[str] = None):
        token = token or settings.get_hf_token()
        self.client = InferenceClient(provider="fal-ai", token=token)

    def transcribe_whisper(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        return_timestamps: bool = True,
    ) -> Dict[str, Any]:
        model = settings.WHISPER_MODEL
        extra_body: Dict[str, Any] = {}
        if return_timestamps:
            extra_body["return_timestamps"] = True
        if language:
            extra_body["generate_kwargs"] = {"language": language}

        logger.info("Calling InferenceClient.automatic_speech_recognition for model=%s", model)
        result = self.client.automatic_speech_recognition(audio_path, model=model, extra_body=extra_body)
        try:
            return dict(result)
        except Exception:
            try:
                return result.__dict__
            except Exception:
                return {"result": str(result)}

    def diarize_pyannote(
        self,
        audio_path: str,
        *,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise RuntimeError("pyannote.audio not installed. Install with: pip install 'pyannote.audio==3.2.0' and ffmpeg.") from e

        if settings.HF_API_TOKEN:
            os.environ.setdefault("HF_HUB_TOKEN", settings.HF_API_TOKEN)
            try:
                from huggingface_hub import login
                login(token=settings.HF_API_TOKEN, add_to_git_credential=False)
            except Exception:
                pass

        pipeline = Pipeline.from_pretrained(settings.DIARIZATION_MODEL)
        pipeline.to(torch.device("cuda"))

        kwargs: Dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization = pipeline(audio_path, **kwargs)
        segments: List[Dict[str, Any]] = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            segments.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(label)})

        return {"raw": str(diarization), "segments": sorted(segments, key=lambda s: s["start"])}
