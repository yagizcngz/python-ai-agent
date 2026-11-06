"""
Unit tests for KeyManager multi-key rotation and quota failover.
"""

from unittest.mock import MagicMock
from ai_agent.core.keys import KeyManager, is_quota_or_rate_limit_error
from ai_agent.core.agent import Agent, AgentEventType
from ai_agent.tools.base import ToolRegistry


def test_is_quota_or_rate_limit_error():
    assert is_quota_or_rate_limit_error(Exception("HTTP 429 Too Many Requests"))
    assert is_quota_or_rate_limit_error(Exception("Rate limit reached for free model daily"))
    assert is_quota_or_rate_limit_error(Exception("Insufficient credits available"))

    custom_err = Exception("Generic error")
    custom_err.status_code = 402
    assert is_quota_or_rate_limit_error(custom_err)

    custom_err_429 = Exception("Generic error")
    custom_err_429.status_code = 429
    assert is_quota_or_rate_limit_error(custom_err_429)

    assert not is_quota_or_rate_limit_error(Exception("Network socket timeout"))
    assert not is_quota_or_rate_limit_error(Exception("SyntaxError in user script"))

    # Unrelated errors that merely contain quota-like words must not burn a key
    bad_request = Exception("Error code: 400 - insufficient context length for this request")
    bad_request.status_code = 400
    assert not is_quota_or_rate_limit_error(bad_request)
    assert not is_quota_or_rate_limit_error(Exception("Model accreditation-402b-instruct not found"))
    assert not is_quota_or_rate_limit_error(Exception("Insufficient permissions to read file"))


def test_key_manager_rotation(tmp_path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("sk-test-key-1\n# comment\n\nsk-test-key-2\nsk-test-key-3\n", encoding="utf-8")

    km = KeyManager(key_file=key_file)
    assert km.total_keys == 3
    assert km.has_multiple_keys is True
    assert km.get_current_key() == "sk-test-key-1"

    # Rotate to key 2
    success, new_key, msg = km.rotate_key()
    assert success is True
    assert new_key == "sk-test-key-2"
    assert km.get_current_key() == "sk-test-key-2"
    assert "2 of 3" in msg

    # Rotate to key 3
    success, new_key, msg = km.rotate_key()
    assert success is True
    assert new_key == "sk-test-key-3"
    assert km.get_current_key() == "sk-test-key-3"
    assert "3 of 3" in msg

    # All exhausted
    success, new_key, msg = km.rotate_key()
    assert success is False
    assert new_key == ""
    assert "All 3 API keys in API_KEYS_OPEN_ROUTER.txt have exhausted their usage limit" in msg


def test_agent_auto_rotates_keys_on_quota_error(tmp_path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("sk-key-a\nsk-key-b\n", encoding="utf-8")
    km = KeyManager(key_file=key_file)

    # First client call throws 429 rate limit, second call succeeds
    mock_client = MagicMock()
    success_resp = MagicMock()
    choice = MagicMock()
    choice.message.content = "All tasks completed successfully."
    choice.message.tool_calls = None
    success_resp.choices = [choice]
    success_resp.usage.prompt_tokens = 50
    success_resp.usage.completion_tokens = 20

    # First client raises 429, rotated client succeeds
    mock_client1 = MagicMock()
    mock_client1.chat.completions.create.side_effect = Exception("Rate limit 429 exceeded for free model daily")

    mock_client2 = MagicMock()
    mock_client2.chat.completions.create.return_value = success_resp

    agent = Agent(
        client=mock_client1,
        model="test-model",
        tools=ToolRegistry([]),
        key_manager=km,
        client_factory=lambda key: mock_client2,
    )

    events = []
    agent.event_handler = lambda e: events.append(e)

    res = agent.run("Hello test")
    assert res == "All tasks completed successfully."
    # Key rotated to key 2
    assert km.current_index == 1
    assert any(e.event_type == AgentEventType.THINKING and "Switched to API key" in str(e.data) for e in events)


def test_key_manager_gemini_provider(tmp_path):
    gemini_key_file = tmp_path / "API_KEYS_GEMINI.txt"
    gemini_key_file.write_text("AIzaSy-gemini-key-1\nAIzaSy-gemini-key-2\n", encoding="utf-8")

    km = KeyManager(key_file=gemini_key_file, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    assert km.total_keys == 2
    assert km.get_current_key() == "AIzaSy-gemini-key-1"

    success, new_key, msg = km.rotate_key()
    assert success is True
    assert new_key == "AIzaSy-gemini-key-2"
    assert km.get_current_key() == "AIzaSy-gemini-key-2"
    assert "2 of 2" in msg
    assert km.source_label == "API_KEYS_GEMINI.txt"


def test_agent_auto_rotates_gemini_keys_on_quota_error(tmp_path):
    gemini_key_file = tmp_path / "API_KEYS_GEMINI.txt"
    gemini_key_file.write_text("AIzaSy-gemini-key-1\nAIzaSy-gemini-key-2\n", encoding="utf-8")
    km = KeyManager(key_file=gemini_key_file, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    mock_client1 = MagicMock()
    mock_client1.chat.completions.create.side_effect = Exception("HTTP 429: Resource has been exhausted (e.g. check quota)")

    mock_client2 = MagicMock()
    success_resp = MagicMock()
    choice = MagicMock()
    choice.message.content = "Gemini response completed."
    choice.message.tool_calls = None
    success_resp.choices = [choice]
    success_resp.usage.prompt_tokens = 40
    success_resp.usage.completion_tokens = 15
    mock_client2.chat.completions.create.return_value = success_resp

    agent = Agent(
        client=mock_client1,
        model="gemini-1.5-flash",
        tools=ToolRegistry([]),
        key_manager=km,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        client_factory=lambda key: mock_client2,
    )

    events = []
    agent.event_handler = lambda e: events.append(e)

    res = agent.run("Perform a task with Gemini")
    assert res == "Gemini response completed."
    assert km.current_index == 1
    assert km.get_current_key() == "AIzaSy-gemini-key-2"
    assert any("Switched to API key 2 of 2" in str(e.data) for e in events)


