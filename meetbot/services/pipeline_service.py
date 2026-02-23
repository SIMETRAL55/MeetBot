from typing import Any, Dict, Optional

from meetbot.services.alignment_service import build_speaker_transcript, format_result_as_json
from meetbot.services.diarization_service import diarize
from meetbot.services.transcription_service import transcribe


def run_pipeline(
    audio_path: str,
    language: Optional[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    whisp = transcribe(audio_path, language=language, use_cache=use_cache, force_refresh=force_refresh)
    dia = diarize(
        audio_path,
        use_cache=use_cache,
        force_refresh=force_refresh,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    merged = build_speaker_transcript(dia["segments"], whisp["segments"])
    final = format_result_as_json(merged, audio_path)
    return {"transcription": whisp, "diarization": dia, "transcript": merged, "output": final}
