"""
Main application orchestration for MeetBot pipeline.

Provides high-level composition of services for end-to-end audio processing:
transcription → diarization → alignment → indexing → RAG queries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .config import settings

logger = logging.getLogger(__name__)


class MeetBotPipeline:
    """
    High-level orchestration for complete MeetBot pipeline.

    Coordinates transcription, diarization, alignment, and output generation.
    """

    def __init__(self):
        """Initialize pipeline with default settings."""
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        audio_path: str,
        output_path: str = "results/output.json",
        backend: Optional[str] = None,
        language: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run complete pipeline on audio file.

        Args:
            audio_path: Path to audio file
            output_path: Where to save result JSON
            backend: Transcription backend ('local' or 'huggingface')
            language: Language hint
            use_cache: Whether to use caching
            force_refresh: Force fresh API calls
            min_speakers: Minimum speaker count
            max_speakers: Maximum speaker count

        Returns:
            dict: Pipeline result with segments and metadata
        """
        results = {}

        try:
            # Import services locally to avoid circular imports
            from .services.transcriber import TranscriberService
            from .services.diarizer import DiarizationService
            from .services.aligner import AlignerService
            from .services.formatters import format_result_as_json
            from .adapters.transcribers import get_transcriber_from_cli_arg

            # 1. Transcription
            self.logger.info("Running transcription...")
            transcriber_adapter = get_transcriber_from_cli_arg(backend_arg=backend)
            transcriber_svc = TranscriberService(transcriber=transcriber_adapter)
            transcription = transcriber_svc.transcribe(
                audio_path,
                language=language,
                use_cache=use_cache,
                force_refresh=force_refresh,
            )
            results["transcription"] = transcription
            self.logger.info(
                f"✓ Transcription: {len(transcription.get('segments', []))} segments"
            )

            # 2. Diarization
            self.logger.info("Running diarization...")
            diarizer_svc = DiarizationService()
            diarization = diarizer_svc.diarize(
                audio_path,
                use_cache=use_cache,
                force_refresh=force_refresh,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            results["diarization"] = diarization
            self.logger.info(
                f"✓ Diarization: {len(diarization.get('segments', []))} segments"
            )

            # 3. Alignment
            self.logger.info("Running alignment...")
            aligner = AlignerService()
            aligned = aligner.build_speaker_transcript(
                diarization.get("segments", []),
                transcription.get("segments", []),
            )
            results["aligned"] = aligned
            self.logger.info(f"✓ Alignment: {len(aligned)} segments")

            # 4. Format output
            final_output = format_result_as_json(aligned, audio_path)
            results["output"] = final_output

            # 5. Save to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✓ Results saved to {output_file}")

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def build_index(
        self,
        transcript_json: str,
        db_root: str = "db",
        embedding_model: str = "./models/all-MiniLM-L6-v2",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Build vector index from transcript for RAG queries.

        Args:
            transcript_json: Path to transcript JSON from run()
            db_root: Root directory for vector databases
            embedding_model: Embedding model name/path
            overwrite: Force rebuild if exists

        Returns:
            Dict with index path and document count
        """
        try:
            import json as _json
            from .services.rag.indexer import RAGIndexer

            self.logger.info("Building RAG index (multilevel: doc+segment+chunk)...")

            # 1. Read transcript JSON segments
            self.logger.info("  1) Reading transcript segments...")
            with open(transcript_json, "r", encoding="utf-8") as fh:
                raw = _json.load(fh)
            segments = raw if isinstance(raw, list) else raw.get("segments", [])
            self.logger.info(f"    ✓ Loaded {len(segments)} segments")

            # 2. Build multilevel vector index (atomic)
            self.logger.info("  2) Building multilevel vector index...")
            collection_name = Path(transcript_json).stem
            db_dir = RAGIndexer.build_multilevel_index_atomic(
                segments=segments,
                persist_root=db_root,
                collection_name=collection_name,
                embedding_model=embedding_model,
                device="cpu",
                job_id=collection_name,
                version=1,
            )
            self.logger.info("    ✓ Multilevel index built")
            self.logger.info(f"✓ Index saved to {db_dir}")

            return {
                "db_dir": db_dir,
                "n_documents": len(segments),
            }

        except Exception as e:
            self.logger.error(f"Index building failed: {e}", exc_info=True)
            raise

    def answer_query(
        self,
        question: str,
        db_dir: str,
        embedding_model: str = "./models/all-MiniLM-L6-v2",
        hf_model: str = "deepseek-ai/DeepSeek-V3.1",
        k: int = 4,
        use_local_llm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Answer question using indexed transcript.

        Args:
            question: User question
            db_dir: Path to vector database (from build_index)
            embedding_model: Embedding model for retrieval
            hf_model: HuggingFace model for generation
            k: Number of documents to retrieve
            use_local_llm: Override use_local_llm setting

        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            from .services.query_service import QueryService

            self.logger.info(f"Answering query: {question[:80]}...")
            query_svc = QueryService()

            result = query_svc.query(
                question=question,
                db_dir=db_dir,
                embedding_model=embedding_model,
                hf_model=hf_model,
                k=k,
                use_local_llm=use_local_llm,
            )

            self.logger.info(f"✓ Answer generated ({len(result['answer'])} chars)")
            return result

        except Exception as e:
            self.logger.error(f"Query failed: {e}", exc_info=True)
            raise


# Convenience function for single-call pipeline
def run_full_pipeline(**kwargs) -> Dict[str, Any]:
    """Convenience function to run full pipeline."""
    pipeline = MeetBotPipeline()
    return pipeline.run(**kwargs)
