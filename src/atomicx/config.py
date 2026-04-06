"""Global configuration for AtomicX services."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Database ──────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://atomicx:atomicx_dev@localhost:5432/atomicx",
        description="Async SQLAlchemy database URL (asyncpg)",
    )
    database_url_sync: str = Field(
        default="postgresql://atomicx:atomicx_dev@localhost:5432/atomicx",
        description="Sync SQLAlchemy database URL (for Alembic)",
    )

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # ── Kafka ─────────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers (comma-separated)",
    )

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector DB URL",
    )

    # ── Exchange ──────────────────────────────────────────────
    binance_ws_url: str = Field(
        default="wss://stream.binance.com:9443",
        description="Binance WebSocket base URL (without endpoint path)",
    )

    # ── Data Sources ──────────────────────────────────────────
    coingecko_api_url: str = Field(
        default="https://api.coingecko.com/api/v3",
        description="CoinGecko API base URL",
    )

    # ── Trading Config ────────────────────────────────────────
    default_symbols: list[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        description="Default trading pairs to track",
    )
    data_poll_interval_seconds: int = Field(
        default=60,
        description="Polling interval for REST-based data sources",
    )

    # ── Logging ───────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Log level")

    # ── LLM / Bedrock ─────────────────────────────────────────
    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS Access Key ID for Bedrock access",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS Secret Access Key for Bedrock access",
    )
    aws_region_name: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock",
    )
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20240620-v1:0",
        description="Bedrock model ID (e.g., Claude 3.5 Sonnet)",
    )
    aws_bearer_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("aws_bearer_token", "AWS_BEARER_TOKEN_BEDROCK"),
        description="Bearer token for Bedrock proxy authentication",
    )
    aws_endpoint_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("aws_endpoint_url", "AWS_ENDPOINT_URL_BEDROCK"),
        description="Custom endpoint URL for Bedrock proxy",
    )

    # ── Anthropic Direct ──────────────────────────────────────
    anthropic_api_key: str | None = Field(
        default=None,
        description="Direct Anthropic API Key (bypasses Bedrock)",
    )

    # ── News & Social APIs ────────────────────────────────────
    news_api_key: str | None = Field(
        default=None,
        description="NewsAPI.org key for global news fallback",
    )
    twitter_bearer_token: str | None = Field(
        default=None,
        description="X (Twitter) API v2 Bearer Token",
    )
    reddit_client_id: str | None = Field(
        default=None,
        description="Reddit API Client ID (PRAW)",
    )
    reddit_client_secret: str | None = Field(
        default=None,
        description="Reddit API Client Secret (PRAW)",
    )
    reddit_user_agent: str = Field(
        default="AtomicX/1.0.0 (by /u/shaonsikder)",
        description="Reddit API User Agent",
    )

    # ── Cost Optimization ─────────────────────────────────────
    news_scan_interval_minutes: int = Field(
        default=5,
        description="How often to scan news feeds (in minutes). Higher = lower API costs. Default: 5 minutes",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}



@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
