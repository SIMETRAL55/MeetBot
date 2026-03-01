"""
Main CLI entry point for MeetBot.

Provides command-line interface for:
- Audio transcription with multi-backend support (transcribe command)
- Speaker diarization and alignment
- Index building for RAG (index command)
- Query answering over indexed transcripts (query command)
- Full pipeline execution (full command)

Supports both backward-compatible positional audio argument and new subcommands.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import settings
from .logging_conf import setup_logging
from .adapters.transcribers import get_transcriber_from_cli_arg
from .services.transcriber import TranscriberService
from .services.diarizer import DiarizationService
from .services.aligner import AlignerService
from .services.formatters import format_result_as_json


logger = logging.getLogger(__name__)


def _convert_audio_to_wav(audio_path: str, logger) -> tuple:
    """Convert audio to WAV if needed. Returns (audio_path, wav_path_or_none)."""
    if audio_path.lower().endswith('.wav'):
        return audio_path, None

    logger.info(f"Converting {audio_path} to WAV format...")
    from .utils.audio import convert_to_wav
    wav_path = convert_to_wav(audio_path, output_dir="temp")
    logger.info(f"✓ Converted to: {wav_path}")
    return wav_path, wav_path


def _cleanup_wav(wav_path, logger):
    """Clean up temporary WAV file if it exists."""
    if wav_path and Path(wav_path).exists():
        try:
            import os
            os.remove(wav_path)
            logger.debug(f"Cleaned up temporary WAV file: {wav_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file {wav_path}: {e}")


def _cleanup_temp_folder(logger, temp_dir="temp"):
    """Clean up entire temporary folder after transcription/diarization."""
    if Path(temp_dir).exists():
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"✓ Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temp folder {temp_dir}: {e}")


def cmd_transcribe(args):
    """Transcribe and diarize audio."""
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("MeetBot Transcription Pipeline Started")
    logger.info(f"Audio file: {args.audio}")

    try:
        audio_path, wav_path = _convert_audio_to_wav(args.audio, logger)

        try:
            # Transcription
            logger.info("1) Transcribing (Whisper)...")
            transcriber_backend = get_transcriber_from_cli_arg(backend_arg=args.backend)
            transcriber_service = TranscriberService(transcriber=transcriber_backend)
            whisp = transcriber_service.transcribe(
                audio_path,
                language=args.language,
                use_cache=args.use_cache,
                force_refresh=args.force_refresh,
            )
            logger.info(
                f"  ✓ {len(whisp.get('segments', []))} transcription segments"
            )

            # Diarization
            logger.info("2) Diarization (Pyannote)...")
            diarizer_service = DiarizationService()
            dia = diarizer_service.diarize(
                audio_path,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                use_cache=args.use_cache,
                force_refresh=args.force_refresh,
            )
            logger.info(f"  ✓ {len(dia.get('segments', []))} speaker segments")

            # Alignment
            logger.info("3) Aligning segments...")
            aligner = AlignerService()
            merged = aligner.build_speaker_transcript(
                dia.get("segments", []),
                whisp.get("segments", []),
            )
            logger.info(f"  ✓ {len(merged)} aligned segments")

            # Format and save
            final = format_result_as_json(merged, args.audio)

            if args.out:
                output_path = Path(args.out)
            else:
                audio_name = Path(args.audio).stem
                output_path = Path("results") / f"{audio_name}.json"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Results saved to: {output_path}")
            print("=" * 60)
            print("TRANSCRIPTION COMPLETED SUCCESSFULLY")
            print(f"Output: {output_path}")
            print("=" * 60)

        finally:
            _cleanup_wav(wav_path, logger)
            _cleanup_temp_folder(logger)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_index(args):
    """Build vector index from transcript."""
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Building vector index...")

    try:
        from .services.prepare_docs import PrepareDocsService
        from .services.indexer import IndexerService

        # Prepare documents
        logger.info("1) Preparing documents...")
        prepare_svc = PrepareDocsService()
        docs, prepared_path = prepare_svc.prepare(args.transcript_json)
        logger.info(f"  ✓ Prepared {len(docs)} documents")

        # Build vector index
        logger.info("2) Building vector index...")
        indexer_svc = IndexerService()
        indexer_svc.build_index(
            str(prepared_path),
            persist_root=args.db_root,
            embedding_model=args.embedding_model,
            overwrite=args.overwrite,
        )

        db_dir = Path(args.db_root) / Path(args.transcript_json).stem
        logger.info(f"✓ Index saved to {db_dir}")
        print("=" * 60)
        print("INDEXING COMPLETED SUCCESSFULLY")
        print(f"Database: {db_dir}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_query(args):
    """Answer question using indexed transcript."""
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Answering query: {args.question[:100]}...")

    try:
        from .services.query_service import QueryService

        query_svc = QueryService()
        result = query_svc.query(
            question=args.question,
            db_dir=args.db_dir,
            embedding_model=args.embedding_model,
            hf_model=args.hf_model,
            k=args.k,
            use_local_llm=args.use_local_llm,
        )

        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["answer"])
        print("\n" + "=" * 60)
        print(f"SOURCES ({len(result['sources'])})")
        print("=" * 60)

        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source['audio_file']} - {source['speaker']}")
            print(f"    Time: {source['start']:.1f}s - {source['end']:.1f}s")
            print(f"    {source['text']}...")

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_create_user(args):
    """Create a new user for the web application."""
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    try:
        from .db.database import init_db, get_session
        from .db.crud import create_user, get_user_by_username
        from .web.auth import hash_password

        init_db()
        SessionLocal = get_session()
        db = SessionLocal()

        try:
            existing = get_user_by_username(db, args.username)
            if existing:
                print(f"Error: User '{args.username}' already exists.")
                sys.exit(1)

            password = args.password
            if not password:
                import getpass
                password = getpass.getpass("Password: ")
                confirm = getpass.getpass("Confirm password: ")
                if password != confirm:
                    print("Error: Passwords do not match.")
                    sys.exit(1)

            hashed = hash_password(password)
            user = create_user(
                db,
                username=args.username,
                password_hash=hashed,
                display_name=args.display_name,
                is_admin=args.admin,
            )
            print(f"✓ User '{user.username}' created successfully (admin={user.is_admin})")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to create user: {e}", exc_info=True)
        sys.exit(1)


def cmd_serve(args):
    """Start the MeetBot web application."""
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting MeetBot web application...")
    try:
        from .web.main import start_server
        start_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="MeetBot: Audio transcription + speaker diarization + RAG indexing"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Transcribe command (also default for backward compatibility)
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe and diarize audio")
    transcribe_parser.add_argument("audio", help="Path to audio file (wav / mp3 / m4a / flac / aac / etc.)")
    transcribe_parser.add_argument("--backend", default=None, choices=["local", "huggingface"],
                                  help="Transcription backend")
    transcribe_parser.add_argument("--language", default=None, help="Language hint for Whisper")
    transcribe_parser.add_argument("--out", default=None, help="Output JSON file")
    transcribe_parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=True)
    transcribe_parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    transcribe_parser.add_argument("--force-refresh", action="store_true")
    transcribe_parser.add_argument("--min-speakers", type=int, default=None)
    transcribe_parser.add_argument("--max-speakers", type=int, default=None)
    transcribe_parser.add_argument("--log-level", default="INFO")
    transcribe_parser.set_defaults(func=cmd_transcribe)

    # Index command
    index_parser = subparsers.add_parser("index", help="Build vector index from transcript")
    index_parser.add_argument("transcript_json", help="Path to transcript JSON")
    index_parser.add_argument("--db-root", default="db", help="Root directory for vector databases")
    index_parser.add_argument("--embedding-model", default="./models/sarashina-embedding-v1-1b",
                             help="Embedding model name/path")
    index_parser.add_argument("--overwrite", action="store_true", help="Force rebuild if exists")
    index_parser.add_argument("--log-level", default="INFO")
    index_parser.set_defaults(func=cmd_index)

    # Query command
    query_parser = subparsers.add_parser("query", help="Answer question using indexed transcript")
    query_parser.add_argument("db_dir", help="Path to vector database")
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument("--embedding-model", default="./models/sarashina-embedding-v1-1b")
    query_parser.add_argument("--hf-model", default="deepseek-ai/DeepSeek-V3.1",
                             help="HuggingFace model for generation")
    query_parser.add_argument("--k", type=int, default=4, help="Number of documents to retrieve")
    query_parser.add_argument("--use-local-llm", action="store_true", help="Use local LLM if available")
    query_parser.add_argument("--log-level", default="INFO")
    query_parser.set_defaults(func=cmd_query)

    # Create user command
    user_parser = subparsers.add_parser("create-user", help="Create a new web application user")
    user_parser.add_argument("username", help="Username for the new user")
    user_parser.add_argument("--password", default=None, help="Password (prompted if not provided)")
    user_parser.add_argument("--display-name", default=None, help="Display name")
    user_parser.add_argument("--admin", action="store_true", help="Grant admin privileges")
    user_parser.add_argument("--log-level", default="INFO")
    user_parser.set_defaults(func=cmd_create_user)

    # Serve command (start web application)
    serve_parser = subparsers.add_parser("serve", help="Start the MeetBot web application")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    serve_parser.add_argument("--log-level", default="INFO")
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    # Handle backward compatibility: if first arg looks like a file and no subcommand, treat as transcribe
    if args.command is None:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Reconstruct as transcribe command
            sys.argv.insert(1, "transcribe")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(0)

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
