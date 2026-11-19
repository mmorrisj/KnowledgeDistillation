"""Runtime settings, read from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tracking_enabled: bool = False
    site_dist: Path = Path("site/dist")
    host: str = "127.0.0.1"
    port: int = 8000
