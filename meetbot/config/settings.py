from typing import ClassVar, Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    HF_API_TOKEN: Optional[str] = Field(None, env="HF_API_TOKEN")
    HF_HUB_TOKEN: Optional[str] = Field(None, env="HF_HUB_TOKEN")
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = Field(None, env="HUGGINGFACEHUB_API_TOKEN")

    WHISPER_MODEL: str = "openai/whisper-large-v3"
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    DIARIZATION_MODEL_REVISION: ClassVar[str] = "main"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def get_hf_token(self) -> Optional[str]:
        return self.HF_API_TOKEN or self.HF_HUB_TOKEN or self.HUGGINGFACEHUB_API_TOKEN


settings = Settings()
