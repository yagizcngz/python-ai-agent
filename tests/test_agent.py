"""
Test suite for Agent ReAct reasoning loop with mocked API completions.
"""

from pathlib import Path
from unittest.mock import MagicMock

from ai_agent.core.agent import Agent, AgentEvent, AgentEventType
from ai_agent.core.memory import ConversationMemory
from ai_agent.tools.base import ToolRegistry
from ai_agent.tools.filesystem import ReadTool, WriteTool


def make_mock_choice(content=None, tool_calls=None):
    choice = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice.message = msg
    return choice


def test_agent_direct_answer(tmp_path: Path):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [make_mock_choice(content="Hello! How can I help you?")]
    mock_response.usage = MagicMock(prompt_tokens=15, completion_tokens=10)
    mock_client.chat.completions.create.return_value = mock_response

    tools = ToolRegistry([ReadTool(tmp_path), WriteTool(tmp_path)])
    agent = Agent(client=mock_client, model="test-model", tools=tools)

    events = []
    agent.event_handler = lambda e: events.append(e)

    answer = agent.run("Hello")
    assert answer == "Hello! How can I help you?"
    assert agent.memory.total_prompt_tokens == 15
    assert agent.memory.total_completion_tokens == 10

    # Ensure answer event was emitted
    answer_events = [e for e in events if e.event_type == AgentEventType.ANSWER]
    assert len(answer_events) == 1
    assert answer_events[0].data == "Hello! How can I help you?"


def test_agent_tool_calling_loop(tmp_path: Path):
    mock_client = MagicMock()

    # Step 1: Model calls Write tool
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "Write"
    tool_call.function.arguments = '{"file_path": "greeting.txt", "content": "Hello file!"}'

    response_1 = MagicMock()
    response_1.choices = [make_mock_choice(content=None, tool_calls=[tool_call])]
    response_1.usage = MagicMock(prompt_tokens=20, completion_tokens=15)

    # Step 2: Model returns synthesized answer
    response_2 = MagicMock()
    response_2.choices = [make_mock_choice(content="I have created greeting.txt for you.")]
    response_2.usage = MagicMock(prompt_tokens=30, completion_tokens=10)

    mock_client.chat.completions.create.side_effect = [response_1, response_2]

    tools = ToolRegistry([ReadTool(tmp_path), WriteTool(tmp_path)])
    agent = Agent(client=mock_client, model="test-model", tools=tools)

    events = []
    agent.event_handler = lambda e: events.append(e)

    result = agent.run("Create greeting.txt with 'Hello file!'")
    assert result == "I have created greeting.txt for you."

    # Check file on disk
    created_file = tmp_path / "greeting.txt"
    assert created_file.exists()
    assert created_file.read_text(encoding="utf-8") == "Hello file!"

    # Verify event stream
    tool_calls = [e for e in events if e.event_type == AgentEventType.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].data["name"] == "Write"


def test_agent_max_steps_guard(tmp_path: Path):
    mock_client = MagicMock()

    # Model continuously requests tool calls
    tool_call = MagicMock()
    tool_call.id = "call_infinite"
    tool_call.function.name = "Read"
    tool_call.function.arguments = '{"file_path": "missing.txt"}'

    response = MagicMock()
    response.choices = [make_mock_choice(content=None, tool_calls=[tool_call])]
    response.usage = None
    mock_client.chat.completions.create.return_value = response

    tools = ToolRegistry([ReadTool(tmp_path)])
    agent = Agent(client=mock_client, model="test-model", tools=tools, max_steps=3)

    result = agent.run("Do infinite work")
    assert "Limit reached: Agent reached maximum steps (3)" in result


def test_extract_tool_or_answer_all_formats():
    from ai_agent.core.agent import extract_tool_or_answer

    available = {"Read", "Write", "Edit", "ListDir", "Bash"}

    # 1. XML style function call
    xml_text = '<function><name>ListDir</name><arguments>{"directory_path":"."}</arguments></function>'
    kind, tool, args = extract_tool_or_answer(xml_text, available)
    assert kind == "tool_call"
    assert tool == "ListDir"

    # 2. Aliased tool name from small model hallucination (ListFilesCount -> ListDir)
    alias_text = '{"name": "ListFilesCount", "arguments": {"directory_path": "."}}'
    kind, tool, args = extract_tool_or_answer(alias_text, available)
    assert kind == "tool_call"
    assert tool == "ListDir"

    # 3. Disguised greeting JSON (Welcome)
    welcome_text = '{"name": "Welcome", "arguments": []}'
    kind, ans, _ = extract_tool_or_answer(welcome_text, available)
    assert kind == "answer"
    assert "AI assistant" in ans

    # 4. Long conversational sentence disguised as tool name
    sentence_text = '{"name": "My name is a large artificial intelligence. How can I help you today?"}'
    kind, ans, _ = extract_tool_or_answer(sentence_text, available)
    assert kind == "answer"
    assert "My name is a large artificial intelligence" in ans

    # 5. Code fence markdown tool call
    fence_text = '```json\n{"name": "read_file", "arguments": {"file_path": "README.md"}}\n```'
    kind, tool, args = extract_tool_or_answer(fence_text, available)
    assert kind == "tool_call"
    assert tool == "Read"


def test_agent_fallback_tool_calling_loop(tmp_path: Path):
    mock_client = MagicMock()

    # Step 1: Model returns tool call in text content rather than tool_calls field
    response_1 = MagicMock()
    response_1.choices = [make_mock_choice(
        content='{"name": "ListFilesCount", "arguments": {"directory_path": "."}}',
        tool_calls=None,
    )]
    response_1.usage = MagicMock(prompt_tokens=20, completion_tokens=15)

    # Step 2: Model returns final answer
    response_2 = MagicMock()
    response_2.choices = [make_mock_choice(
        content="There are 0 files in this directory.",
        tool_calls=None,
    )]
    response_2.usage = MagicMock(prompt_tokens=30, completion_tokens=10)

    mock_client.chat.completions.create.side_effect = [response_1, response_2]

    from ai_agent.tools.filesystem import ListDirTool
    tools = ToolRegistry([ListDirTool(tmp_path)])
    agent = Agent(client=mock_client, model="local-small-model", tools=tools)

    events = []
    agent.event_handler = lambda e: events.append(e)

    result = agent.run("can you count the files under this folder ?")
    assert result == "There are 0 files in this directory."

    tool_call_events = [e for e in events if e.event_type == AgentEventType.TOOL_CALL]
    assert len(tool_call_events) == 1
    assert tool_call_events[0].data["name"] == "ListDir"


def test_custom_provider_validation_and_usage():
    from ai_agent.config import is_openrouter_url
    from ai_agent.core.client import fetch_account_usage, validate_model_id

    # 1. URL detection
    assert is_openrouter_url("https://openrouter.ai/api/v1") is True
    assert is_openrouter_url("https://generativelanguage.googleapis.com/v1beta/openai/") is False
    assert is_openrouter_url("https://api.groq.com/openai/v1") is False

    # 2. Model validation allows custom models for external providers
    valid, res = validate_model_id("gemini-1.5-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    assert valid is True
    assert res == "gemini-1.5-flash"

    valid, res = validate_model_id("llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1")
    assert valid is True
    assert res == "llama-3.3-70b-versatile"

    # 3. Account usage endpoint returns None for custom providers
    usage = fetch_account_usage(api_key="any-custom-key", base_url="https://api.groq.com/openai/v1")
    assert usage is None



def test_agent_waits_and_retries_same_key_on_429(tmp_path: Path, monkeypatch):
    import ai_agent.core.agent as agent_mod
    sleeps = []
    monkeypatch.setattr(agent_mod.time, "sleep", lambda s: sleeps.append(s))

    rate_err = Exception("Rate limit exceeded: 15 requests per minute")
    rate_err.status_code = 429
    ok = MagicMock()
    ok.choices = [make_mock_choice(content="done")]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [rate_err, ok]
    key_manager = MagicMock()

    agent = Agent(client=mock_client, model="m", tools=ToolRegistry([]), key_manager=key_manager)
    assert agent.run("hi") == "done"
    assert sleeps == [2.0]
    key_manager.rotate_key.assert_not_called()


def test_memory_trims_old_tool_outputs(monkeypatch):
    import ai_agent.core.memory as memory_mod
    monkeypatch.setattr(memory_mod, "MAX_CONTEXT_CHARS", 1000)
    memory = ConversationMemory(system_prompt="sys")
    for i in range(10):
        memory.add_assistant_message(tool_calls=[{"id": f"c{i}", "function": {"name": "Read", "arguments": "{}"}}])
        memory.add_tool_result(f"c{i}", "x" * 500)

    messages = memory.get_messages()
    assert len(messages) == 21
    assert messages[2]["content"] == memory_mod.TRIMMED_TOOL_OUTPUT
    assert messages[-1]["content"] == "x" * 500
    assert sum(len(str(m.get("content") or "")) for m in messages) <= 1000 + 500 * 3


def test_console_renders_tool_output_with_brackets():
    from rich.console import Console
    from ai_agent.ui.console import RichAgentConsole

    console = Console(record=True, width=80)
    RichAgentConsole(console=console).handle_event(
        AgentEvent(AgentEventType.TOOL_RESULT, 1, {"name": "Read", "result": "x = a[/b]\ny: list[int]"})
    )
    text = console.export_text()
    assert "a[/b]" in text and "list[int]" in text
