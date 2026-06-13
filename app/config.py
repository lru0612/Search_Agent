from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"

    tavily_api_key: str = ""
    jina_api_key: str = ""

    max_steps: int = 10
    max_reflect_rounds: int = 2
    max_clarify_rounds: int = 2
    token_budget: int = 120_000
    context_token_limit: int = 45_000

    tool_timeout_s: float = 20.0
    page_content_max_chars: int = 24_000  # 约 8k tokens
    traces_dir: str = "traces"


@lru_cache
def get_settings() -> Settings:
    return Settings()
