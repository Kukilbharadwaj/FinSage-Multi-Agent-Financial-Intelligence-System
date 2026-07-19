# config/settings.py
# Load environment variables using pydantic-settings.
# Every agent imports the `settings` object from here.

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key from https://console.groq.com",
    )
    MCP_ENABLED: bool = Field(
        default=True,
        description="Enable MCP tool calls from the agent pipeline.",
    )
    MCP_TIMEOUT_SECONDS: int = Field(
        default=20,
        description="Timeout (seconds) for each MCP tool call.",
    )

    # MCP_SERVER_URL was removed in v4. Tools are now reached over FastMCP's
    # in-memory transport inside this process, so there is no server URL,
    # no port, and no SSE endpoint to configure.

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings object — import this everywhere
settings = Settings()
