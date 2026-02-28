"""Audio processing utilities for MeetBot."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None


def convert_to_wav(
    input_path: str,
    output_dir: str = "temp",
    output_sample_rate: int = 16000,
    output_channels: int = 1,
) -> str:
    """
    Convert audio file to 16-bit PCM WAV format.

    Converts any audio format to WAV with standard speech processing parameters:
    - Sample rate: 16 kHz (standard for Whisper and Pyannote)
    - Channels: Mono (1)
    - Bit depth: 16-bit PCM

    Args:
        input_path: Path to input audio file (MP3, M4A, FLAC, etc.)
        output_dir: Directory to save converted WAV file (default: "temp")
        output_sample_rate: Target sample rate in Hz (default: 16000)
        output_channels: Target number of channels (default: 1 = mono)

    Returns:
        str: Path to converted WAV file

    Raises:
        RuntimeError: If conversion fails (pydub not installed, unsupported format, etc.)
    """
    if AudioSegment is None:
        raise RuntimeError(
            "pydub not installed. Install with: pip install pydub\n"
            "Also ensure ffmpeg is installed: apt-get install ffmpeg"
        )

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (input_path.stem + ".wav")
    input_format = input_path.suffix.lstrip(".").lower()

    logger.info(
        f"Converting {input_path} to WAV: {output_sample_rate}Hz, {output_channels}ch"
    )

    try:
        audio = AudioSegment.from_file(input_path, format=input_format)
        audio = audio.set_frame_rate(output_sample_rate).set_channels(output_channels)
        audio.export(output_path, format="wav")
        logger.info(f"✓ Converted to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        raise RuntimeError(f"Failed to convert {input_path} to WAV: {e}") from e
