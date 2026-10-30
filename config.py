import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    transcription_model: str = os.getenv("TRANSCRIPTION_MODEL", "base")
    transcription_device: str = os.getenv("TRANSCRIPTION_DEVICE", "auto")
    transcription_compute_type: str = os.getenv("TRANSCRIPTION_COMPUTE_TYPE", "auto")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "500"))
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")


SETTINGS = Settings()
