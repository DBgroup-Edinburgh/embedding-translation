"""Runtime settings driven by environment variables and .env files.

Anything that depends on the deployment environment (API keys, paths, cluster
endpoints) goes through `Settings`, never through ad-hoc os.environ lookups.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ETRANS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # API keys
    openai_api_key: str | None = None
    mistral_api_key: str | None = None
    wandb_api_key: str | None = None

    # Paths
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./.cache")
    output_dir: Path = Path("./output")

    # VectorBenchmark — where it caches embeddings + which HF repo it pulls from
    hf_repo: str = "DB-Edinburgh/VectorBenchmark"
    vectorbench_embedding_dir: Path = Path("./.cache/embeddings")

    # Logging
    log_level: str = "INFO"
    verbose: bool = False


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton. Cheap to call from anywhere."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
