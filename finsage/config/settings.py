# config/settings.py
# Load environment variables using pydantic-settings.
# Every agent imports the `settings` object from here.

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = Field(..., description="Groq API key from https://console.groq.com")
    MCP_ENABLED: bool = Field(
        default=True,
        description="Enable MCP tool calls from agent pipeline when available.",
    )
    MCP_SERVER_URL: str = Field(
        default="http://localhost:8001/sse",
        description="FinSage MCP server SSE endpoint.",
    )
    MCP_TIMEOUT_SECONDS: int = Field(
        default=20,
        description="Timeout (seconds) for each MCP tool call.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings object — import this everywhere
settings = Settings()
