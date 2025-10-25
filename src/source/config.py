# config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import ClassVar, Optional
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Accept any of these common HF env var names 
    HF_API_TOKEN: Optional[str] = Field(None, env="HF_API_TOKEN")
    HF_HUB_TOKEN: Optional[str] = Field(None, env="HF_HUB_TOKEN")
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = Field(None, env="HUGGINGFACEHUB_API_TOKEN")

    # other app defaults
    WHISPER_MODEL: str = "openai/whisper-large-v3"
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    DIARIZATION_MODEL_REVISION: ClassVar[str] = "main"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",   # IMPORTANT: allow unknown env keys instead of failing
    }

    def get_hf_token(self) -> Optional[str]:
        return self.HF_API_TOKEN or self.HF_HUB_TOKEN or self.HUGGINGFACEHUB_API_TOKEN

settings = Settings()
