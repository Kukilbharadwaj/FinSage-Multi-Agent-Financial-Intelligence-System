# config/settings.py
# Load environment variables using pydantic-settings.
# Every agent imports the `settings` object from here.

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = Field(..., description="Groq API key from https://console.groq.com")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings object — import this everywhere
settings = Settings()
