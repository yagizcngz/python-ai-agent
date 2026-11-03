"""
Configuration and environment management for AI Agent.
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project root is the parent directory of this module's root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default workspace directory: AI_AGENT_WORKSPACE if set, else current working directory
WORKSPACE_DIR = Path(os.getenv("AI_AGENT_WORKSPACE", default=str(Path.cwd()))).resolve()

# Default OpenRouter API endpoints
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"

BASE_URL = os.getenv("OPENROUTER_BASE_URL", default=DEFAULT_OPENROUTER_BASE_URL)

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", default="inclusionai/ling-3.0-flash-sante:free")
DEFAULT_LOCAL_MODEL = "qwen2.5-coder:1.5b"

RECOMMENDED_FREE_MODELS: List[str] = [
    "inclusionai/ling-3.0-flash-sante:free",
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-2.5-2.6b:free",
    "google/gemma-4-26b-a4b-it:free",
    "poolside/laguna-xs-2.1:free",
]

POPULAR_LOCAL_MODELS: List[str] = [
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
    "llama3.2:1b",
    "llama3.2:3b",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "mistral:7b",
    "codellama:7b",
]

DEFAULT_MAX_STEPS = 25
DEFAULT_SHELL_TIMEOUT = 30
DEFAULT_API_TIMEOUT = 30.0
MAX_TOOL_OUTPUT_CHARS = 4000
# ~30k tokens; old tool outputs are trimmed past this so small free models don't overflow
MAX_CONTEXT_CHARS = int(os.getenv("AI_AGENT_MAX_CONTEXT_CHARS", "120000"))


def get_api_key(allow_fallback: bool = True) -> str:
    """
    Retrieve API key from environment variable or fallback text file.
    Preserves backwards compatibility with API_KEYS_OPEN_ROUTER.txt.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip()

    if not allow_fallback:
        return ""

    search_paths = [
        PROJECT_ROOT / "API_KEYS_OPEN_ROUTER.txt",
        PROJECT_ROOT / "app" / "API_KEYS_OPEN_ROUTER.txt",
    ]
    for path in search_paths:
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            return line
            except Exception:
                continue
    return ""


def is_local_url(url: str) -> bool:
    """Check if the provided base URL targets a local loopback server."""
    return "localhost" in url or "127.0.0.1" in url or "0.0.0.0" in url


def is_openrouter_url(url: Optional[str] = None) -> bool:
    """Check if the provided base URL targets OpenRouter."""
    target = url or BASE_URL
    return "openrouter.ai" in target if target else True


def get_provider_description(url: Optional[str] = None) -> str:
    """Return a human-friendly provider badge and known limits description."""
    target = (url or BASE_URL).lower()
    if is_local_url(target):
        return "Unlimited (Local)"
    if "openrouter.ai" in target:
        return "OpenRouter"
    if "googleapis.com" in target:
        return "Gemini Free (1,500 RPD / 15 RPM)"
    if "groq.com" in target:
        return "Groq Free (30 RPM)"
    if "deepseek.com" in target:
        return "DeepSeek (Token Metered)"
    if "openai.com" in target:
        return "OpenAI Direct"
    return "Custom API"

