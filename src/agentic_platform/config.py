from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    agent_planner: str = "rules"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    agent_max_steps: int = 5
    agent_max_retries: int = 1
    agent_timeout_seconds: float = 20.0


settings = Settings()
