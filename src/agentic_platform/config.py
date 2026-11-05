from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    agent_runtime_backend: str = "langgraph"
    agent_planner: str = "rules"
    agent_openai_strategy: str = "tool_calling"
    openai_api_key: str | None = None
    agent_api_key: str | None = None
    agent_rate_limit_per_minute: int = 60
    openai_model: str = "gpt-4.1-mini"
    agent_max_steps: int = 5
    agent_max_retries: int = 1
    agent_timeout_seconds: float = 20.0
    agent_database_path: str = "./data/agentic_platform.db"
    agent_memory_max_messages: int = 20
    agent_semantic_memory_enabled: bool = False
    agent_semantic_memory_path: str = "./data/semantic_memory"
    agent_semantic_memory_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    agent_semantic_memory_top_k: int = 3
    agent_semantic_memory_max_items: int = 100
    agent_semantic_memory_ttl_days: int = 0
    agent_semantic_memory_duplicate_distance: float = 0.08


settings = Settings()
