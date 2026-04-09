"""Typed environment-based configuration for Skynet components."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


@dataclass(frozen=True)
class RedisConfig:
    host: str = field(default_factory=lambda: _env("REDIS_HOST", "redis.redis.svc"))
    port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    db: int = field(default_factory=lambda: _env_int("REDIS_DB", 0))

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class QdrantConfig:
    host: str = field(default_factory=lambda: _env("QDRANT_HOST", "qdrant.qdrant.svc"))
    port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class OllamaConfig:
    url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://100.64.0.3:11434"))
    embed_model: str = field(default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 512))


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    api_url: str = field(default_factory=lambda: _env("LLM_API_URL", "https://openrouter.ai/api/v1"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "anthropic/claude-sonnet-4"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "openai/text-embedding-3-small"))


@dataclass(frozen=True)
class MatrixConfig:
    homeserver_url: str = field(default_factory=lambda: _env("MATRIX_HOMESERVER_URL", "http://conduwuit.matrix.svc:6167"))
    bot_token: str = field(default_factory=lambda: _env("MATRIX_BOT_TOKEN"))
    room_id: str = field(default_factory=lambda: _env("MATRIX_ROOM_ID"))
    alerts_room_id: str = field(default_factory=lambda: _env("MATRIX_ALERTS_ROOM_ID"))
    feed_room_id: str = field(default_factory=lambda: _env("MATRIX_FEED_ROOM_ID"))


@dataclass(frozen=True)
class OTelConfig:
    endpoint: str = field(default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT"))
    service_name: str = field(default_factory=lambda: _env("OTEL_SERVICE_NAME", "skynet"))


@dataclass(frozen=True)
class SkynetConfig:
    """Aggregated configuration loaded from environment variables."""

    redis: RedisConfig = field(default_factory=RedisConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    otel: OTelConfig = field(default_factory=OTelConfig)
