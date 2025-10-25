# hf_client.py
# from lib2to3.pgen2 import token
import logging
from pathlib import Path
import mimetypes
from typing import Optional, Dict, Any, List

from huggingface_hub import InferenceClient
from config import settings
import os
from huggingface_hub import login
import torch


logger = logging.getLogger(__name__)


class HFInferenceClient:
    """
    Thin wrapper:
     - automatic_speech_recognition via huggingface_hub.InferenceClient for Whisper
     - local pyannote Pipeline use for speaker diarization (downloads model and runs locally).
    """
    def __init__(self, token: str = settings.HF_API_TOKEN):
        self.client = InferenceClient(provider="fal-ai", token=token)
    # -------------------------
    # Whisper ASR (InferenceClient)
    # -------------------------
    def transcribe_whisper(self, audio_path: str, *, language: Optional[str] = None,
                           return_timestamps: bool = True) -> Dict[str, Any]:
        """
        Use InferenceClient.automatic_speech_recognition to transcribe audio.
        - audio_path can be a path to a file.
        - pass model-specific args via extra_body.
        Returns a dict-like response from the inference client.
        """
        model = settings.WHISPER_MODEL
        extra_body: Dict[str, Any] = {}
        if return_timestamps:
            extra_body["return_timestamps"] = True
        if language:
            extra_body["generate_kwargs"] = {"language": language}

        logger.info("Calling InferenceClient.automatic_speech_recognition for model=%s", model)
        result = self.client.automatic_speech_recognition(audio_path, model=model, extra_body=extra_body)
        print(result)
        try:
            return dict(result)
        except Exception:
            try:
                return result.__dict__
            except Exception:
                return {"result": str(result)}

    # -------------------------
    # Pyannote diarization (local pipeline)
    # -------------------------
    def diarize_pyannote(self, audio_path: str,
                         *, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> Dict[str, Any]:
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise RuntimeError(
                "pyannote.audio not installed. Install with: pip install 'pyannote.audio==3.2.0' and ffmpeg."
            ) from e

        model = settings.DIARIZATION_MODEL
        revision = settings.DIARIZATION_MODEL_REVISION

        # --- Authenticate for model download ---
        if settings.HF_API_TOKEN:
            os.environ.setdefault("HF_HUB_TOKEN", settings.HF_API_TOKEN)
            try:
                from huggingface_hub import login
                login(token=settings.HF_API_TOKEN, add_to_git_credential=False)
            except Exception:
                pass 

        logger.info("Loading Pyannote pipeline %s (revision=%s)...", model, revision)
        pipeline = Pipeline.from_pretrained(model) 
        pipeline.to(torch.device("cuda"))

        # Set min/max speakers
        kwargs: Dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        print("==========DIARIZATION PARAMETERS==============")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
            
        logger.info("Running diarization pipeline on %s with kwargs=%s", audio_path, kwargs)
        diarization = pipeline(audio_path, **kwargs)

        # Format segments
        segments: List[Dict[str, Any]] = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(label)
            })

        return {"raw": str(diarization), "segments": sorted(segments, key=lambda s: s["start"])}
