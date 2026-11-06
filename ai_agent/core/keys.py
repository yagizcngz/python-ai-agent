"""
API key rotation and multi-key quota management.
Supports automatic failover across multiple keys in API_KEYS_OPEN_ROUTER.txt.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from ..config import PROJECT_ROOT


def is_quota_or_rate_limit_error(e: Exception) -> bool:
    """Check if an exception indicates a rate limit or exhausted quota."""
    status_code = getattr(e, "status_code", None)
    if status_code in (402, 429):
        return True
    if isinstance(status_code, int):
        # A real HTTP error with another status (400 bad request, 404 unknown model, 500...)
        # is not a quota problem; rotating would wrongly mark a good key as exhausted.
        return False

    msg = str(e).lower()
    if re.search(r"\b(429|402)\b", msg):
        return True
    quota_phrases = [
        "rate limit",
        "rate_limit",
        "quota",
        "insufficient credit",
        "insufficient_quota",
        "daily limit",
        "usage limit",
        "free model daily",
    ]
    return any(phrase in msg for phrase in quota_phrases)


class KeyManager:
    """
    Manages multiple OpenRouter API keys with automatic failover and rotation.
    Loads keys from API_KEYS_OPEN_ROUTER.txt and active environment variables.
    """

    def __init__(
        self,
        key_file: Optional[Path] = None,
        keys: Optional[List[str]] = None,
        base_url: Optional[str] = None,
    ):
        self.base_url = base_url
        self.key_file = key_file
        self.source_label = key_file.name if key_file else "API_KEYS_OPEN_ROUTER.txt"
        if keys is not None:
            self.keys: List[str] = list(keys)
        else:
            self.keys = self._load_keys()
        self.current_index: int = 0
        self.exhausted_keys: set[str] = set()

    def _determine_search_paths(self) -> List[Path]:
        """Determine key file search paths based on configured endpoint."""
        if self.key_file:
            return [self.key_file]

        files: List[Path] = []
        url = (self.base_url or os.getenv("OPENROUTER_BASE_URL", "")).lower()

        if "generativelanguage.googleapis.com" in url or "google" in url:
            files.extend([
                PROJECT_ROOT / "API_KEYS_GEMINI.txt",
                PROJECT_ROOT / "app" / "API_KEYS_GEMINI.txt",
            ])
            self.source_label = "API_KEYS_GEMINI.txt"
        elif "api.groq.com" in url or "groq" in url:
            files.extend([
                PROJECT_ROOT / "API_KEYS_GROQ.txt",
                PROJECT_ROOT / "app" / "API_KEYS_GROQ.txt",
            ])
            self.source_label = "API_KEYS_GROQ.txt"
        elif "api.deepseek.com" in url or "deepseek" in url:
            files.extend([
                PROJECT_ROOT / "API_KEYS_DEEPSEEK.txt",
                PROJECT_ROOT / "app" / "API_KEYS_DEEPSEEK.txt",
            ])
            self.source_label = "API_KEYS_DEEPSEEK.txt"

        # General fallbacks
        files.extend([
            PROJECT_ROOT / "API_KEYS.txt",
            PROJECT_ROOT / "app" / "API_KEYS.txt",
            PROJECT_ROOT / "API_KEYS_OPEN_ROUTER.txt",
            PROJECT_ROOT / "app" / "API_KEYS_OPEN_ROUTER.txt",
        ])
        return files

    def _load_keys(self) -> List[str]:
        """Load unique, non-empty keys from files and environment."""
        keys: List[str] = []

        search_paths = self._determine_search_paths()

        for path in search_paths:
            if path and path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            clean = line.strip()
                            if clean and not clean.startswith("#") and clean not in keys:
                                keys.append(clean)
                except Exception:
                    pass

        # Also append current env key if set and not already present (only when not loading explicit key_file)
        if not self.key_file:
            url = (self.base_url or os.getenv("OPENROUTER_BASE_URL", "")).lower()
            if "generativelanguage.googleapis.com" in url or "google" in url:
                env_key = os.getenv("GEMINI_API_KEY", "").strip()
            elif "api.groq.com" in url or "groq" in url:
                env_key = os.getenv("GROQ_API_KEY", "").strip()
            elif "api.deepseek.com" in url or "deepseek" in url:
                env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            else:
                env_key = os.getenv("OPENROUTER_API_KEY", "").strip()

            if env_key and env_key not in keys and not env_key.startswith("ollama"):
                keys.insert(0, env_key)

        return keys

    @property
    def total_keys(self) -> int:
        return len(self.keys)

    @property
    def has_multiple_keys(self) -> bool:
        return len(self.keys) > 1

    def get_current_key(self) -> str:
        """Return the currently active API key."""
        if not self.keys:
            return os.getenv("OPENROUTER_API_KEY", "").strip()
        if self.current_index >= len(self.keys):
            self.current_index = 0
        return self.keys[self.current_index]

    def rotate_key(self) -> Tuple[bool, str, str]:
        """
        Advance to the next unexhausted API key.
        Returns: (success: bool, new_key: str, message: str)
        """
        if not self.keys:
            return (
                False,
                "",
                f"No API keys found in {self.source_label} or environment variables.",
            )

        current = self.get_current_key()
        if current:
            self.exhausted_keys.add(current)

        # Search for the next unexhausted key
        for offset in range(1, len(self.keys) + 1):
            next_idx = (self.current_index + offset) % len(self.keys)
            candidate = self.keys[next_idx]
            if candidate not in self.exhausted_keys:
                self.current_index = next_idx
                os.environ["OPENROUTER_API_KEY"] = candidate
                url = (self.base_url or os.getenv("OPENROUTER_BASE_URL", "")).lower()
                if "generativelanguage.googleapis.com" in url or "google" in url:
                    os.environ["GEMINI_API_KEY"] = candidate
                elif "api.groq.com" in url or "groq" in url:
                    os.environ["GROQ_API_KEY"] = candidate
                elif "api.deepseek.com" in url or "deepseek" in url:
                    os.environ["DEEPSEEK_API_KEY"] = candidate

                masked = f"{candidate[:8]}...{candidate[-4:]}" if len(candidate) > 12 else "key"
                msg = f"Switched to API key {self.current_index + 1} of {len(self.keys)} ({masked})."
                return True, candidate, msg

        # All keys in the file have been exhausted
        label = "API_KEYS_OPEN_ROUTER.txt" if self.source_label.endswith("keys.txt") else self.source_label
        return (
            False,
            "",
            f"All {len(self.keys)} API keys in {label} have exhausted their usage limit.",
        )
