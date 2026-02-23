import argparse

from meetbot.services.pipeline_service import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe + Diarize audio")
    parser.add_argument("audio", help="Path to audio file (wav / mp3 / m4a)")
    parser.add_argument("--language", default=None)
    parser.add_argument("--use-cache", default=True, action="store_true")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    result = run_pipeline(
        args.audio,
        language=args.language,
        use_cache=args.use_cache,
        force_refresh=args.force_refresh,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print(f"ASR segments: {len(result['transcription']['segments'])}")
    print(f"Diarization segments: {len(result['diarization']['segments'])}")
    print("=== FINAL OUTPUT ===")
    print(result["output"])


if __name__ == "__main__":
    main()
