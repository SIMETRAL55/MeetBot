# asr_diarization/utils/audio_utils.py
from pydub import AudioSegment
from pathlib import Path

def convert_to_wav(input_path: str, output_dir: str = "temp") -> str:
    """Convert audio to 16-bit PCM WAV (supported by Whisper API)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / (Path(input_path).stem + ".wav")
    # print(Path(input_path).suffix.replace('.', ''))
    audio = AudioSegment.from_file(input_path, format=Path(input_path).suffix.replace('.', ''))
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return str(output_path)
