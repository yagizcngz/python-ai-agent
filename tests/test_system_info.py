import io
import json
import urllib.request
from unittest.mock import patch

from ai_agent.core.system_info import (
    check_local_model_installed,
    evaluate_specs_for_model,
    get_model_spec_requirements,
    get_system_specs,
)


def test_get_system_specs():
    specs = get_system_specs()
    assert "total_ram_gb" in specs
    assert "avail_ram_gb" in specs
    assert "free_disk_gb" in specs
    assert specs["total_ram_gb"] > 0
    assert specs["free_disk_gb"] > 0


def test_get_model_spec_requirements():
    req_1b = get_model_spec_requirements("qwen2.5-coder:1.5b")
    assert req_1b["min_ram_gb"] == 4.0
    assert req_1b["min_disk_gb"] == 2.0

    req_8b = get_model_spec_requirements("llama3.1:8b")
    assert req_8b["min_ram_gb"] == 8.0
    assert req_8b["min_disk_gb"] == 6.0

    req_32b = get_model_spec_requirements("deepseek-coder:33b")
    assert req_32b["min_ram_gb"] == 32.0


def test_evaluate_specs_for_model():
    eval_res = evaluate_specs_for_model("qwen2.5-coder:1.5b")
    assert "is_compatible" in eval_res
    assert "verdict" in eval_res
    assert "ram_pass" in eval_res
    assert "disk_pass" in eval_res


def test_check_local_model_installed_offline():
    # Attempting to query an unreachable local port should return ollama_offline
    installed, reason, models = check_local_model_installed(
        "qwen2.5-coder:1.5b", base_url="http://127.0.0.1:59999", timeout=0.2
    )
    assert installed is False
    assert reason == "ollama_offline"
    assert models == []


def test_check_local_model_installed_mocked():
    mock_payload = {
        "models": [
            {"name": "qwen2.5-coder:1.5b"},
            {"name": "llama3.1:8b:latest"},
        ]
    }
    mock_response = io.BytesIO(json.dumps(mock_payload).encode("utf-8"))

    with patch.object(urllib.request, "urlopen", return_value=mock_response):
        installed, reason, models = check_local_model_installed("qwen2.5-coder:1.5b")
        assert installed is True
        assert reason == "ready"
        assert len(models) == 2

    mock_response_missing = io.BytesIO(json.dumps(mock_payload).encode("utf-8"))
    with patch.object(urllib.request, "urlopen", return_value=mock_response_missing):
        installed, reason, models = check_local_model_installed("mistral:7b")
        assert installed is False
        assert reason == "model_missing"
