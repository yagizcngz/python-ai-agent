"""
OpenAI / OpenRouter API client creation, model discovery, and recovery helpers.
"""

import sys
from typing import List, Optional
from openai import OpenAI
from ..config import (
    BASE_URL,
    DEFAULT_API_TIMEOUT,
    DEFAULT_LOCAL_BASE_URL,
    PROJECT_ROOT,
    RECOMMENDED_FREE_MODELS,
    get_api_key,
    is_local_url,
    is_openrouter_url,
)


def create_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    is_local: bool = False,
    timeout: float = DEFAULT_API_TIMEOUT,
) -> OpenAI:
    """
    Create and configure an OpenAI client instance.
    Supports OpenRouter, local Ollama/LM Studio servers, or any OpenAI-compatible API.
    """
    effective_base_url = base_url or (DEFAULT_LOCAL_BASE_URL if is_local else BASE_URL)
    effective_api_key = api_key or get_api_key()

    # For local loopback endpoints (Ollama, LM Studio), any dummy key is accepted
    if not effective_api_key and is_local_url(effective_base_url):
        effective_api_key = "ollama-local"

    if not effective_api_key:
        raise ValueError(
            "API key is not set. Please configure OPENROUTER_API_KEY (or your provider's API key) "
            "in your environment, add it to a .env file, or pass --local to connect to a local server."
        )

    return OpenAI(api_key=effective_api_key, base_url=effective_base_url, timeout=timeout)


def fetch_models(client: OpenAI, only_free: bool = True) -> List[str]:
    """Fetch active models from the endpoint."""
    try:
        models = client.models.list()
        all_ids = [m.id for m in models.data]
        if only_free:
            free_ids = [m_id for m_id in all_ids if ":free" in m_id]
            if free_ids:
                return free_ids
        return all_ids
    except Exception as e:
        print(f"Notice: Failed to fetch live models from API ({e}). Using defaults.", file=sys.stderr)
        return list(RECOMMENDED_FREE_MODELS)


def get_model_suggestions(current_model: str) -> List[str]:
    """Return alternative free models excluding the current one."""
    return [m for m in RECOMMENDED_FREE_MODELS if m != current_model]


def fetch_account_usage(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[dict]:
    """
    Fetch live remaining limits and usage metrics from OpenRouter key endpoint.
    Returns None if offline, using local models, custom non-OpenRouter providers, or if request fails.
    """
    if not is_openrouter_url(base_url):
        return None

    key = api_key or get_api_key()
    if not key or key.startswith("ollama"):
        return None

    import json
    import time
    import urllib.request

    try:
        url = f"https://openrouter.ai/api/v1/key?_={int(time.time() * 1000)}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "python-ai-agent",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )
        with urllib.request.urlopen(req, timeout=5.0) as res:
            payload = json.loads(res.read().decode("utf-8"))
            return payload.get("data")
    except Exception:
        return None


MODELS_CACHE_FILE = PROJECT_ROOT / ".cache" / "openrouter_models_cache.json"
MODELS_CACHE_TTL = 86400.0  # 24 hours

_MODEL_CACHE: Optional[set[str]] = None
_FREE_TOOL_MODELS_CACHE: Optional[List[str]] = None
_ALL_FREE_MODELS_CACHE: Optional[List[str]] = None


def _load_disk_cache() -> bool:
    """Load cached model lists from local disk. Returns True if valid and unexpired."""
    global _MODEL_CACHE, _FREE_TOOL_MODELS_CACHE, _ALL_FREE_MODELS_CACHE
    if not MODELS_CACHE_FILE.is_file():
        return False

    import json
    import time
    try:
        with open(MODELS_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        cached_all = data.get("all_models", [])
        cached_free_tool = data.get("free_tool_models", [])
        cached_all_free = data.get("all_free_models", [])
        ts = data.get("timestamp", 0.0)

        if cached_all:
            _MODEL_CACHE = set(cached_all)
        if cached_free_tool:
            _FREE_TOOL_MODELS_CACHE = cached_free_tool
        if cached_all_free:
            _ALL_FREE_MODELS_CACHE = cached_all_free

        # Cache is fresh if within TTL
        if time.time() - ts < MODELS_CACHE_TTL and _MODEL_CACHE:
            return True
    except Exception:
        pass
    return False


def _save_disk_cache() -> None:
    """Save model lists to local disk cache."""
    import json
    import time
    try:
        MODELS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "all_models": list(_MODEL_CACHE) if _MODEL_CACHE else [],
            "free_tool_models": _FREE_TOOL_MODELS_CACHE or [],
            "all_free_models": _ALL_FREE_MODELS_CACHE or [],
        }
        with open(MODELS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _fetch_openrouter_models_metadata(force_refresh: bool = False) -> None:
    """Fetch and categorize live OpenRouter models with local disk caching."""
    global _MODEL_CACHE, _FREE_TOOL_MODELS_CACHE, _ALL_FREE_MODELS_CACHE
    if _MODEL_CACHE is not None and not force_refresh:
        return

    # Check local disk cache first unless explicit refresh requested
    if not force_refresh and _load_disk_cache():
        return

    import json
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "python-ai-agent"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as res:
            data = json.loads(res.read().decode("utf-8")).get("data", [])
            if data:
                _MODEL_CACHE = {m["id"] for m in data if "id" in m}
                
                # Filter models that are free AND have native 'tools' support in supported_parameters
                tool_models = [
                    m["id"] for m in data
                    if ":free" in m.get("id", "") and "tools" in m.get("supported_parameters", [])
                ]
                if tool_models:
                    _FREE_TOOL_MODELS_CACHE = tool_models

                all_free = [m["id"] for m in data if ":free" in m.get("id", "")]
                if all_free:
                    _ALL_FREE_MODELS_CACHE = sorted(all_free)

                _save_disk_cache()
                return
    except Exception:
        pass

    # If network call failed, try disk cache even if older than TTL
    if _MODEL_CACHE is None:
        _load_disk_cache()

    if _MODEL_CACHE is None:
        _MODEL_CACHE = set(RECOMMENDED_FREE_MODELS)
    if _FREE_TOOL_MODELS_CACHE is None:
        _FREE_TOOL_MODELS_CACHE = list(RECOMMENDED_FREE_MODELS)
    if _ALL_FREE_MODELS_CACHE is None:
        _ALL_FREE_MODELS_CACHE = list(RECOMMENDED_FREE_MODELS)


def get_all_valid_models(force_refresh: bool = False) -> set[str]:
    """Fetch and cache the full set of valid OpenRouter model IDs."""
    _fetch_openrouter_models_metadata(force_refresh=force_refresh)
    return _MODEL_CACHE or set(RECOMMENDED_FREE_MODELS)


def get_recommended_free_models(force_refresh: bool = False) -> List[str]:
    """
    Return the curated top recommended free models that have verified tool support.
    Verifies they exist in active catalog when online; falls back to RECOMMENDED_FREE_MODELS if offline.
    """
    _fetch_openrouter_models_metadata(force_refresh=force_refresh)
    if _FREE_TOOL_MODELS_CACHE:
        curated_active = [m for m in RECOMMENDED_FREE_MODELS if m in _FREE_TOOL_MODELS_CACHE]
        if curated_active:
            return curated_active
        return _FREE_TOOL_MODELS_CACHE[:5]
    return list(RECOMMENDED_FREE_MODELS)


def get_all_free_models(force_refresh: bool = False) -> List[str]:
    """Fetch and return all active free models (:free) from OpenRouter."""
    _fetch_openrouter_models_metadata(force_refresh=force_refresh)
    return _ALL_FREE_MODELS_CACHE or list(RECOMMENDED_FREE_MODELS)


def validate_model_id(
    model_id: str,
    is_local: bool = False,
    base_url: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Check if a model exists on OpenRouter, custom provider, or local server.
    Returns (is_valid, resolved_model_or_error_message).
    """
    cleaned = model_id.strip()
    if not cleaned:
        return False, "Model ID cannot be empty."

    # If it's a local Ollama model
    if is_local or cleaned.startswith("ollama/"):
        return True, cleaned[7:] if cleaned.startswith("ollama/") else cleaned

    # If it's a custom non-OpenRouter provider (Google AI Studio, Groq, OpenAI direct, etc.)
    if not is_openrouter_url(base_url):
        return True, cleaned

    valid_models = get_all_valid_models()

    if cleaned in valid_models:
        return True, cleaned

    # Check without :free suffix if present, or with :free suffix
    if cleaned.endswith(":free") and cleaned[:-5] in valid_models:
        return True, cleaned
    if f"{cleaned}:free" in valid_models:
        return True, cleaned

    # Case-insensitive match check
    lowered = {m.lower(): m for m in valid_models}
    if cleaned.lower() in lowered:
        return True, lowered[cleaned.lower()]

    base_lower = cleaned[:-5].lower() if cleaned.endswith(":free") else cleaned.lower()
    if base_lower in lowered:
        return True, lowered[base_lower]

    return False, f"Model '{cleaned}' does not exist on OpenRouter."

