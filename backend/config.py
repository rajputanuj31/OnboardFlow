from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    github_token: str
    openai_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    eval_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        extra="ignore"
    )

settings = Settings()