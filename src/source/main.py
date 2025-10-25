# main.py
import argparse
from transcribe import transcribe
from diarize import diarize
from align import build_speaker_transcript, format_result_as_json
import json

def main():
    parser = argparse.ArgumentParser(description="Transcribe + Diarize audio using Hugging Face Inference (with caching)")
    parser.add_argument("audio", help="Path to audio file (wav / mp3 / m4a)")
    parser.add_argument("--language", help="Language hint for Whisper (e.g. 'en')", default=None)
    parser.add_argument("--out", help="Output json file", default="diarized_transcript.json")
    parser.add_argument("--use-cache", help="Use cached HF raw responses when available", default=True, action="store_true")
    parser.add_argument("--no-cache", help="Do not use cache; forces fresh HF calls", dest="use_cache", action="store_false")
    parser.add_argument("--force-refresh", help="Force fresh HF call and overwrite cache", action="store_true")
    parser.add_argument("--min-speakers", help="Minumum number of speaker in the audio recording", type=int, default=None)
    parser.add_argument("--max-speakers", help="Maximum number of speaker in the audio recording", type=int, default=None)
    args = parser.parse_args()

    audio_path = args.audio

    print("1) Transcribing (Whisper)...")
    whisp = transcribe(audio_path, language=args.language, use_cache=args.use_cache, force_refresh=args.force_refresh)
    # print(whisp["chunks"])
    print(f"  -> Normalized ASR segments: {len(whisp['segments'])}")
    print(f"  -> Whisper raw from_cache={whisp.get('from_cache')}, cache_path={whisp.get('cache_path')}")

    print("2) Diarization (Pyannote)...")
    dia = diarize(audio_path, use_cache=args.use_cache, force_refresh=args.force_refresh, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    print(f"  -> Diarization segments: {len(dia['segments'])}")
    print(f"  -> Diarization raw from_cache={dia.get('from_cache')}, cache_path={dia.get('cache_path')}")

    print("3) Aligning segments...")
    merged = build_speaker_transcript(dia["segments"], whisp["segments"])
    # print(whisp["segments"])
    # print(dia["segments"])
    final = format_result_as_json(merged, audio_path)
    print("========FINAL OUTPUT==========")
    print(final)
    # print(json.dumps(final, indent=4))

if __name__ == "__main__":
    main()
