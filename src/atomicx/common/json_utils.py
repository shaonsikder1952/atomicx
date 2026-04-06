"""Robust JSON extraction from LLM responses.

Handles common issues:
- Markdown code blocks (```json ... ```)
- Extra narrative text before/after JSON
- Malformed JSON with common errors
- Multiple JSON objects in response
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger


def extract_json_from_llm_text(
    text: str,
    expected_keys: list[str] | None = None,
    default: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Safely extract JSON from LLM response text.

    Args:
        text: Raw LLM response text
        expected_keys: Optional list of keys that must be present
        default: Default dict to return if extraction fails

    Returns:
        Parsed JSON dict or default/None if extraction fails
    """
    if not text or not isinstance(text, str):
        return default

    # Strategy 1: Try direct parse (cleanest case)
    try:
        result = json.loads(text.strip())
        if _validate_keys(result, expected_keys):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Remove markdown code blocks
    try:
        # Remove ```json or ``` wrappers
        cleaned = re.sub(r'```(?:json)?\s*', '', text)
        cleaned = re.sub(r'\s*```', '', cleaned)
        result = json.loads(cleaned.strip())
        if _validate_keys(result, expected_keys):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 3: Extract first complete JSON object
    try:
        # Find first { and matching }
        start = text.find('{')
        if start == -1:
            return default

        # Count braces to find matching close
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_text = text[start:i+1]
                    result = json.loads(json_text)
                    if _validate_keys(result, expected_keys):
                        return result
                    break
    except (json.JSONDecodeError, TypeError, IndexError):
        pass

    # Strategy 4: Aggressive cleaning - remove common text patterns
    try:
        # Remove common preambles
        cleaned = re.sub(r'^.*?(?=\{)', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'\}.*?$', '}', cleaned, flags=re.DOTALL)
        result = json.loads(cleaned.strip())
        if _validate_keys(result, expected_keys):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 5: Try to fix common JSON errors
    try:
        # Replace single quotes with double quotes
        fixed = text.replace("'", '"')
        # Remove trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        result = json.loads(fixed)
        if _validate_keys(result, expected_keys):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 6: Extract with simple regex (last resort)
    try:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            result = json.loads(match.group())
            if _validate_keys(result, expected_keys):
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # All strategies failed
    logger.warning(f"Failed to extract JSON from LLM text (tried 6 strategies): {text[:100]}...")
    return default


def _validate_keys(data: Any, expected_keys: list[str] | None) -> bool:
    """Check if parsed data contains expected keys."""
    if not isinstance(data, dict):
        return False

    if expected_keys is None:
        return True

    return all(key in data for key in expected_keys)


def safe_json_loads(
    text: str,
    default: Any = None,
    log_errors: bool = True,
) -> Any:
    """Safe json.loads with fallback.

    Args:
        text: JSON text to parse
        default: Default value if parsing fails
        log_errors: Whether to log parsing errors

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        if log_errors:
            logger.debug(f"JSON parse error: {e} - text: {text[:100]}")
        return default
