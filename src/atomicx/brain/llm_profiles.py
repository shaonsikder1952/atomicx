"""LLM Profile Manager with Failover & Auth Rotation.

Extracted from OpenClaw's battle-tested failover patterns.
Implements two-tier resilience:
1. Auth profile rotation (same provider, different keys)
2. Model fallback chain (different providers)

Prevents single point of failure in LLM infrastructure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"  # Local fallback


@dataclass
class LLMProfile:
    """A single LLM authentication profile."""

    name: str
    provider: LLMProvider
    priority: int  # Lower = higher priority (0 = primary)

    # Provider-specific config
    api_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_bearer_token: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # Model configuration
    model_id: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 8000
    temperature: float = 0.7

    # Health tracking
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0

    def is_available(self) -> bool:
        """Check if profile is available (not in cooldown)."""
        return time.time() >= self.cooldown_until

    def mark_success(self) -> None:
        """Record successful API call."""
        self.last_success_time = time.time()
        self.consecutive_failures = 0
        self.cooldown_until = 0.0

    def mark_failure(self) -> None:
        """Record failed API call and enter exponential cooldown."""
        self.last_failure_time = time.time()
        self.consecutive_failures += 1

        # Exponential cooldown: 1m → 5m → 25m → 1h (max)
        cooldown_seconds = min(60 * (5 ** (self.consecutive_failures - 1)), 3600)
        self.cooldown_until = time.time() + cooldown_seconds

        logger.warning(
            f"[LLM-FAILOVER] Profile '{self.name}' failed {self.consecutive_failures} times. "
            f"Cooldown: {cooldown_seconds}s"
        )


class LLMProfileManager:
    """Manages LLM profile failover with auth rotation."""

    def __init__(self, profiles: list[LLMProfile]):
        """Initialize with list of profiles (sorted by priority)."""
        self.profiles = sorted(profiles, key=lambda p: p.priority)
        self.logger = logger.bind(module="brain.llm_profiles")

        self.logger.info(
            f"[LLM-FAILOVER] Initialized with {len(self.profiles)} profiles: "
            + ", ".join(f"{p.name} ({p.provider.value})" for p in self.profiles)
        )

    def get_active_profile(self) -> Optional[LLMProfile]:
        """Get the highest-priority available profile.

        Returns None if all profiles are in cooldown (emergency situation).
        """
        for profile in self.profiles:
            if profile.is_available():
                return profile

        # All profiles in cooldown - return least-cooled profile
        self.logger.error("[LLM-FAILOVER] ALL PROFILES IN COOLDOWN - using least-cooled")
        return min(self.profiles, key=lambda p: p.cooldown_until)

    def get_client(self, profile: LLMProfile) -> tuple[Any, str]:
        """Create LLM client for given profile.

        Returns:
            (client, provider_name) tuple
        """
        if profile.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=profile.api_key)
                return client, "anthropic"
            except Exception as e:
                self.logger.error(f"[LLM-FAILOVER] Failed to create Anthropic client: {e}")
                profile.mark_failure()
                raise

        elif profile.provider == LLMProvider.BEDROCK:
            try:
                import boto3

                client_kwargs = {
                    "service_name": "bedrock-runtime",
                    "region_name": profile.aws_region or "us-east-1",
                }

                if profile.aws_bearer_token:
                    # Proxy mode
                    client_kwargs["aws_access_key_id"] = "PROXY"
                    client_kwargs["aws_secret_access_key"] = "PROXY"
                elif profile.aws_access_key_id and profile.aws_secret_access_key:
                    # Direct AWS credentials
                    client_kwargs["aws_access_key_id"] = profile.aws_access_key_id
                    client_kwargs["aws_secret_access_key"] = profile.aws_secret_access_key

                client = boto3.client(**client_kwargs)
                return client, "bedrock"
            except Exception as e:
                self.logger.error(f"[LLM-FAILOVER] Failed to create Bedrock client: {e}")
                profile.mark_failure()
                raise

        elif profile.provider == LLMProvider.OLLAMA:
            try:
                # Local Ollama - use direct HTTP client
                import httpx
                client = httpx.AsyncClient(base_url=profile.ollama_base_url, timeout=60.0)
                return client, "ollama"
            except Exception as e:
                self.logger.error(f"[LLM-FAILOVER] Failed to create Ollama client: {e}")
                profile.mark_failure()
                raise

        else:
            raise ValueError(f"Unsupported provider: {profile.provider}")

    async def call_with_failover(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> tuple[str, str]:
        """Call LLM with automatic failover on errors.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            (response_text, profile_name) tuple

        Raises:
            RuntimeError: If all profiles fail
        """
        last_error = None

        # Try each available profile in priority order
        for attempt, profile in enumerate(self.profiles):
            if not profile.is_available():
                self.logger.debug(f"[LLM-FAILOVER] Skipping profile '{profile.name}' (in cooldown)")
                continue

            try:
                self.logger.info(
                    f"[LLM-FAILOVER] Attempt {attempt + 1}/{len(self.profiles)}: Using profile '{profile.name}' ({profile.provider.value})"
                )

                client, provider = self.get_client(profile)

                # Call provider-specific API
                if provider == "anthropic":
                    response = await client.messages.create(
                        model=profile.model_id,
                        max_tokens=profile.max_tokens,
                        temperature=profile.temperature,
                        system=system_prompt or "",
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                    text = response.content[0].text

                elif provider == "bedrock":
                    import json
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": profile.max_tokens,
                        "temperature": profile.temperature,
                        "system": system_prompt or "",
                        "messages": [{"role": "user", "content": prompt}],
                        **kwargs
                    })

                    response = client.invoke_model(
                        modelId=profile.model_id,
                        body=body
                    )

                    response_body = json.loads(response['body'].read())
                    text = response_body['content'][0]['text']

                elif provider == "ollama":
                    # Ollama API format
                    response = await client.post(
                        "/api/generate",
                        json={
                            "model": profile.model_id,
                            "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                            "stream": False,
                            "options": {
                                "temperature": profile.temperature,
                                "num_predict": profile.max_tokens,
                            }
                        }
                    )
                    response.raise_for_status()
                    text = response.json()["response"]

                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                # Success!
                profile.mark_success()
                self.logger.success(
                    f"[LLM-FAILOVER] ✓ Success with profile '{profile.name}' ({profile.provider.value})"
                )
                return text, profile.name

            except Exception as e:
                last_error = e
                profile.mark_failure()
                self.logger.error(
                    f"[LLM-FAILOVER] ✗ Failed with profile '{profile.name}': {e}"
                )
                continue

        # All profiles failed
        error_msg = f"All {len(self.profiles)} LLM profiles failed. Last error: {last_error}"
        self.logger.critical(f"[LLM-FAILOVER] {error_msg}")
        raise RuntimeError(error_msg)

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all profiles."""
        return {
            "total_profiles": len(self.profiles),
            "available_profiles": sum(1 for p in self.profiles if p.is_available()),
            "profiles": [
                {
                    "name": p.name,
                    "provider": p.provider.value,
                    "priority": p.priority,
                    "available": p.is_available(),
                    "consecutive_failures": p.consecutive_failures,
                    "cooldown_seconds": max(0, p.cooldown_until - time.time()),
                }
                for p in self.profiles
            ]
        }


def create_default_profiles() -> list[LLMProfile]:
    """Create default profile configuration from environment.

    Priority order:
    1. Anthropic API (primary key)
    2. Anthropic API (backup key if available)
    3. AWS Bedrock (fallback)
    4. Local Ollama (emergency)
    """
    from atomicx.config import get_settings
    settings = get_settings()

    profiles = []

    # Profile 1: Anthropic primary
    if settings.anthropic_api_key:
        profiles.append(LLMProfile(
            name="anthropic_primary",
            provider=LLMProvider.ANTHROPIC,
            priority=0,
            api_key=settings.anthropic_api_key,
            model_id="claude-sonnet-4-5-20250929",
        ))

    # Profile 2: Anthropic backup (if ANTHROPIC_BACKUP_API_KEY is set)
    import os
    backup_key = os.getenv("ANTHROPIC_BACKUP_API_KEY")
    if backup_key:
        profiles.append(LLMProfile(
            name="anthropic_backup",
            provider=LLMProvider.ANTHROPIC,
            priority=1,
            api_key=backup_key,
            model_id="claude-sonnet-4-5-20250929",
        ))

    # Profile 3: Bedrock fallback
    if settings.aws_access_key_id or settings.aws_bearer_token:
        profiles.append(LLMProfile(
            name="bedrock_fallback",
            provider=LLMProvider.BEDROCK,
            priority=2,
            aws_region=settings.aws_region_name,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_bearer_token=settings.aws_bearer_token,
            model_id=settings.bedrock_model_id or "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ))

    # Profile 4: Local Ollama (emergency)
    # Only add if user has explicitly set OLLAMA_ENABLED=true
    if os.getenv("OLLAMA_ENABLED", "false").lower() == "true":
        profiles.append(LLMProfile(
            name="ollama_emergency",
            provider=LLMProvider.OLLAMA,
            priority=3,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model_id=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            max_tokens=4000,  # Lower for local models
        ))

    if not profiles:
        raise RuntimeError(
            "No LLM profiles configured. Set ANTHROPIC_API_KEY or AWS credentials in .env"
        )

    logger.info(f"[LLM-FAILOVER] Created {len(profiles)} profiles: {[p.name for p in profiles]}")
    return profiles
