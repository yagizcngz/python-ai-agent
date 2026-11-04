"""
System specs inspection and local model hardware compatibility checks.
"""

import ctypes
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple


def get_system_specs() -> Dict[str, any]:
    """Retrieve system RAM and disk availability across drives in GB."""
    free_disk_gb = 0.0
    drives = {}
    try:
        free_disk_gb = round(shutil.disk_usage(Path.home()).free / (1024**3), 1)
    except Exception:
        try:
            free_disk_gb = round(shutil.disk_usage(".").free / (1024**3), 1)
        except Exception:
            free_disk_gb = 10.0

    if sys.platform == "win32":
        import string

        for letter in string.ascii_uppercase:
            drive_path = f"{letter}:\\"
            if Path(drive_path).exists():
                try:
                    usage = shutil.disk_usage(drive_path)
                    drives[letter] = {
                        "total_gb": round(usage.total / (1024**3), 1),
                        "free_gb": round(usage.free / (1024**3), 1),
                    }
                except Exception:
                    pass

    max_free_disk_gb = max([d["free_gb"] for d in drives.values()], default=free_disk_gb)

    total_ram_gb = 8.0
    avail_ram_gb = 4.0

    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024**3), 1)
                avail_ram_gb = round(stat.ullAvailPhys / (1024**3), 1)
        except Exception:
            pass

    return {
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "free_disk_gb": free_disk_gb,
        "drives": drives,
        "max_free_disk_gb": max_free_disk_gb,
    }


def get_model_spec_requirements(model_name: str) -> Dict[str, any]:
    """Estimate hardware requirements based on model parameter size."""
    m_lower = model_name.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b\b", m_lower)

    if match:
        size = float(match.group(1))
        if size <= 2.5:
            return {
                "min_ram_gb": 4.0,
                "min_disk_gb": 2.0,
                "approx_download": "1.1 GB",
                "tier": "Lightweight (Suitable for any modern PC)",
            }
        elif size <= 4.5:
            return {
                "min_ram_gb": 6.0,
                "min_disk_gb": 3.5,
                "approx_download": "2.3 GB",
                "tier": "Medium (Requires 6+ GB RAM)",
            }
        elif size <= 9.0:
            return {
                "min_ram_gb": 8.0,
                "min_disk_gb": 6.0,
                "approx_download": "4.7 GB",
                "tier": "Standard (Requires 8+ GB RAM)",
            }
        elif size <= 16.0:
            return {
                "min_ram_gb": 16.0,
                "min_disk_gb": 12.0,
                "approx_download": "9.0 GB",
                "tier": "Heavy (Requires 16+ GB RAM)",
            }
        else:
            return {
                "min_ram_gb": 32.0,
                "min_disk_gb": 25.0,
                "approx_download": "20.0 GB",
                "tier": "Workstation (Requires 32+ GB RAM & dedicated GPU)",
            }

    return {
        "min_ram_gb": 4.0,
        "min_disk_gb": 2.0,
        "approx_download": "1.5 GB",
        "tier": "General Model",
    }


def evaluate_specs_for_model(model_name: str) -> Dict[str, any]:
    """Compare user's PC hardware specs against model requirements."""
    specs = get_system_specs()
    reqs = get_model_spec_requirements(model_name)

    ram_pass = specs["total_ram_gb"] >= (reqs["min_ram_gb"] * 0.85)
    disk_pass = (specs["free_disk_gb"] >= reqs["min_disk_gb"]) or (specs.get("max_free_disk_gb", 0) >= reqs["min_disk_gb"])
    is_compatible = ram_pass and disk_pass

    if is_compatible:
        verdict = "Your PC meets the hardware specs for this model."
    elif not ram_pass and not disk_pass:
        verdict = "Insufficient RAM and free disk space for this model."
    elif not ram_pass:
        verdict = f"RAM ({specs['total_ram_gb']} GB) is below the recommended {reqs['min_ram_gb']} GB."
    else:
        verdict = f"Free disk space ({specs['free_disk_gb']} GB) is below {reqs['min_disk_gb']} GB required."

    return {
        **specs,
        **reqs,
        "ram_pass": ram_pass,
        "disk_pass": disk_pass,
        "is_compatible": is_compatible,
        "verdict": verdict,
    }


_OLLAMA_CACHE_TS: float = 0.0
_OLLAMA_CACHE_DATA: List[str] = []
_OLLAMA_CACHE_TTL: float = 30.0  # 30 seconds cache


def check_local_model_installed(
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: float = 1.5,
    force_refresh: bool = False,
) -> Tuple[bool, str, List[str]]:
    """
    Check whether Ollama is active and whether the target model is installed.
    Uses a 30s in-memory cache to prevent socket delays when rendering UI dialogs.
    Returns: (is_installed, reason_code, list_of_installed_models)
    reason_code can be: 'ready', 'ollama_offline', or 'model_missing'
    """
    global _OLLAMA_CACHE_TS, _OLLAMA_CACHE_DATA
    import time

    now = time.time()
    installed_models = []

    if not force_refresh and (now - _OLLAMA_CACHE_TS) < _OLLAMA_CACHE_TTL and _OLLAMA_CACHE_DATA:
        installed_models = _OLLAMA_CACHE_DATA
    else:
        endpoint = f"{base_url.rstrip('/')}/api/tags"
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": "python-ai-agent"})
            with urllib.request.urlopen(req, timeout=timeout) as res:
                data = json.loads(res.read().decode("utf-8"))
                installed_models = [m.get("name", "") for m in data.get("models", [])]
                _OLLAMA_CACHE_DATA = installed_models
                _OLLAMA_CACHE_TS = now
        except Exception:
            if not _OLLAMA_CACHE_DATA:
                return False, "ollama_offline", []
            installed_models = _OLLAMA_CACHE_DATA

    target = model_name.lower().strip()
    if not target:
        return True if installed_models else False, "ready" if installed_models else "model_missing", installed_models

    # Normalize model comparison (e.g. 'qwen2.5-coder:1.5b' matches 'qwen2.5-coder:1.5b:latest')
    for inst in installed_models:
        inst_lower = inst.lower()
        if target == inst_lower or f"{target}:latest" == inst_lower or target == inst_lower.split(":")[0]:
            return True, "ready", installed_models

    return False, "model_missing", installed_models
