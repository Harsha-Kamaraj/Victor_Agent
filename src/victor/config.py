"""Configuration, loaded from the environment and ``.env``.

API keys keep their conventional names (``GROQ_API_KEY``) so they interoperate
with other tooling; everything Victor-specific is prefixed ``VICTOR_``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import Paths

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def _env(*names: str) -> AliasChoices:
    return AliasChoices(*names)


class Settings(BaseSettings):
    """Everything Victor needs to know before it starts."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- credentials ------------------------------------------------------
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=_env("GROQ_API_KEY", "VICTOR_GROQ_API_KEY")
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=_env("GEMINI_API_KEY", "VICTOR_GEMINI_API_KEY")
    )
    github_token: SecretStr | None = Field(
        default=None, validation_alias=_env("GITHUB_TOKEN", "VICTOR_GITHUB_TOKEN")
    )

    # --- behaviour --------------------------------------------------------
    data_dir: Path = Field(
        default=Path.home() / ".victor", validation_alias=_env("VICTOR_DATA_DIR")
    )
    log_level: LogLevel = Field(default="INFO", validation_alias=_env("VICTOR_LOG_LEVEL"))
    dry_run: bool = Field(default=False, validation_alias=_env("VICTOR_DRY_RUN"))
    confirm_destructive: bool = Field(
        default=True, validation_alias=_env("VICTOR_CONFIRM_DESTRUCTIVE")
    )
    strict_free_tier: bool = Field(
        default=True, validation_alias=_env("VICTOR_STRICT_FREE_TIER")
    )

    # --- model overrides --------------------------------------------------
    text_model: str | None = Field(default=None, validation_alias=_env("VICTOR_TEXT_MODEL"))
    vision_model: str | None = Field(default=None, validation_alias=_env("VICTOR_VISION_MODEL"))
    stt_model: str | None = Field(default=None, validation_alias=_env("VICTOR_STT_MODEL"))

    @field_validator("data_dir", mode="before")
    @classmethod
    def _blank_means_default(cls, v: object) -> object:
        # An empty VICTOR_DATA_DIR= line in .env should mean "use the default",
        # not "use the current directory".
        if isinstance(v, str) and not v.strip():
            return Path.home() / ".victor"
        return v

    @field_validator("groq_api_key", "gemini_api_key", "github_token", mode="before")
    @classmethod
    def _blank_means_absent(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("data_dir", mode="after")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    @property
    def paths(self) -> Paths:
        return Paths(self.data_dir)

    def secret(self, name: str) -> str | None:
        """Read a credential by field name, unwrapping the SecretStr."""
        value = getattr(self, name, None)
        return value.get_secret_value() if isinstance(value, SecretStr) else None

    def has(self, name: str) -> bool:
        """True if the named credential is configured and non-empty."""
        return bool(self.secret(name))


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached; call :func:`reset_settings` in tests."""
    return Settings()


def reset_settings() -> None:
    """Drop the cached settings so the next read re-reads the environment."""
    get_settings.cache_clear()
