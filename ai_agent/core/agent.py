"""
Agent reasoning loop (ReAct pattern) with step guards and event emission.
"""

from dataclasses import dataclass
from enum import Enum
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid
from openai import OpenAI

from ..config import DEFAULT_MAX_STEPS
from ..tools.base import TOOL_ALIASES, ToolRegistry
from .client import get_model_suggestions
from .keys import KeyManager, is_quota_or_rate_limit_error
from .memory import ConversationMemory


def extract_tool_or_answer(text: str, available_tools: Set[str]) -> Tuple[str, Any, Any]:
    """
    Examines assistant response text for tool calls or disguised conversational output.
    Returns:
        ("tool_call", tool_name, arguments_str)
        or
        ("answer", cleaned_answer_text, None)
    """
    text = (text or "").strip()
    if not text:
        return ("answer", "", None)

    fn_name = None
    fn_args: Any = {}

    # 1. XML style <function><name>...</name><arguments>...</arguments></function>
    m = re.search(r'<function>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</function>', text, re.DOTALL)
    if m:
        fn_name = m.group(1).strip()
        raw_args = m.group(2).strip()
        try:
            import yaml
            fn_args = yaml.safe_load(raw_args) or {}
        except Exception:
            fn_args = raw_args

    # 2. <tool_call>...</tool_call>
    if not fn_name:
        m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if m:
            try:
                import yaml
                d = yaml.safe_load(m.group(1).strip())
                if isinstance(d, dict) and "name" in d:
                    fn_name = d.get("name")
                    fn_args = d.get("arguments", {})
            except Exception:
                pass

    # 3. Code fence ```(?:json|yaml)? ... ```
    if not fn_name:
        m = re.search(r'```(?:json|yaml)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            try:
                import yaml
                d = yaml.safe_load(m.group(1))
                if isinstance(d, dict) and "name" in d:
                    fn_name = d.get("name")
                    fn_args = d.get("arguments", {})
            except Exception:
                pass

    # 4. Raw JSON / YAML object
    if not fn_name and text.startswith('{') and text.endswith('}'):
        try:
            import yaml
            d = yaml.safe_load(text)
            if isinstance(d, dict) and "name" in d:
                fn_name = d.get("name")
                fn_args = d.get("arguments", {})
        except Exception:
            pass

    if not fn_name or not isinstance(fn_name, str):
        return ("answer", text, None)

    # Resolve aliases
    resolved_name = TOOL_ALIASES.get(fn_name.lower().replace("-", "_"), fn_name)
    tool_map = {t.lower(): t for t in available_tools}
    if resolved_name.lower() in tool_map:
        matched_name = tool_map[resolved_name.lower()]
        args_str = json.dumps(fn_args) if isinstance(fn_args, (dict, list)) else str(fn_args)
        return ("tool_call", matched_name, args_str)

    # Conversational text disguised as JSON
    if fn_name.lower() in ("welcome", "greeting", "introduce", "introduction", "hello", "hi"):
        return ("answer", "Hello! I am your AI assistant. How can I help you today?", None)

    if len(fn_name) > 20 or " " in fn_name:
        return ("answer", fn_name, None)

    if isinstance(fn_args, dict):
        for k in ("message", "text", "response", "content"):
            if k in fn_args and isinstance(fn_args[k], str):
                return ("answer", fn_args[k], None)

    # Fallback to plain text
    return ("answer", text, None)


RATE_LIMIT_WAITS = (2.0, 6.0)  # seconds between same-key retries on HTTP 429
MAX_RETRY_AFTER = 30.0


def _retry_after_seconds(e: Exception) -> Optional[float]:
    """Read a numeric Retry-After header from an API error, capped so the agent never stalls long."""
    try:
        value = float(e.response.headers.get("retry-after"))  # type: ignore[attr-defined]
    except Exception:
        return None
    return min(max(value, 0.0), MAX_RETRY_AFTER)


class AgentEventType(str, Enum):
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"
    ERROR = "error"
    STEP = "step"


@dataclass
class AgentEvent:
    event_type: AgentEventType
    step: int
    data: Any


class Agent:
    """Autonomous agent engine executing the ReAct (Reason-Act-Observe) loop."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: ToolRegistry,
        memory: Optional[ConversationMemory] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        event_handler: Optional[Callable[[AgentEvent], None]] = None,
        key_manager: Optional[KeyManager] = None,
        is_local: bool = False,
        base_url: Optional[str] = None,
        client_factory: Optional[Callable[[str], Any]] = None,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.memory = memory or ConversationMemory()
        self.max_steps = max_steps
        self.event_handler = event_handler
        self.key_manager = key_manager
        self.is_local = is_local
        self.base_url = base_url
        self.client_factory = client_factory
        self.current_step = 0

    def _emit(self, event_type: AgentEventType, data: Any) -> None:
        """Emit an event if a handler is registered."""
        if self.event_handler:
            try:
                self.event_handler(AgentEvent(event_type=event_type, step=self.current_step, data=data))
            except Exception:
                pass

    def _tool_usage_summary(self) -> str:
        """Count tool calls in history, e.g. 'Read (2x), Bash (1x)'."""
        counts: Dict[str, int] = {}
        for msg in self.memory.messages:
            for tc in msg.get("tool_calls") or []:
                name = tc.get("function", {}).get("name", "Tool")
                counts[name] = counts.get(name, 0) + 1
        return ", ".join(f"{k} ({v}x)" for k, v in counts.items()) if counts else "None"

    def run(self, prompt: str) -> str:
        """
        Execute an autonomous task to completion.
        Maintains memory across multiple turns if run repeatedly.
        """
        self.memory.add_user_message(prompt)
        self.current_step = 0
        openai_tools = self.tools.get_openai_tools()

        while True:
            self.current_step += 1
            if self.current_step > self.max_steps:
                msg = (
                    f"Limit reached: Agent reached maximum steps ({self.max_steps}). Stopping loop.\n\n"
                    f"**Summary of progress:**\n"
                    f"- Steps executed: {self.max_steps}/{self.max_steps}\n"
                    f"- Tools executed: {self._tool_usage_summary()}\n"
                    f"- Status: Partial progress preserved in workspace.\n\n"
                    f"*Tip: Prompt 'Continue' to have the agent resume working from where it stopped.*"
                )
                self._emit(AgentEventType.ERROR, msg)
                return msg

            self._emit(AgentEventType.STEP, self.current_step)
            self._emit(AgentEventType.THINKING, f"Consulting model '{self.model}'...")

            chat = None
            waits_left = len(RATE_LIMIT_WAITS)
            while True:
                try:
                    chat = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.memory.get_messages(),
                        tools=openai_tools if openai_tools else None,
                    )
                    break
                except Exception as e:
                    # Per-minute limits clear quickly: wait and retry the same key before burning it.
                    is_daily = any(s in str(e).lower() for s in ("daily", "per day", "perday"))
                    if not self.is_local and waits_left and not is_daily and getattr(e, "status_code", None) == 429:
                        delay = _retry_after_seconds(e) or RATE_LIMIT_WAITS[-waits_left]
                        waits_left -= 1
                        self._emit(AgentEventType.THINKING, f"Rate limited, waiting {delay:g}s before retrying...")
                        time.sleep(delay)
                        continue
                    if self.key_manager and is_quota_or_rate_limit_error(e):
                        waits_left = len(RATE_LIMIT_WAITS)
                        success, new_key, msg = self.key_manager.rotate_key()
                        if success:
                            self._emit(AgentEventType.THINKING, f"{msg} Retrying request...")
                            if self.client_factory:
                                self.client = self.client_factory(new_key)
                            else:
                                from .client import create_client
                                self.client = create_client(
                                    api_key=new_key,
                                    base_url=self.base_url,
                                    is_local=self.is_local,
                                )
                            continue
                        else:
                            err_msg = (
                                f"API Limit Reached: {msg}\n\n"
                                f"**Summary of progress before running out of requests:**\n"
                                f"- Steps completed: {self.current_step - 1} steps\n"
                                f"- Tools executed: {self._tool_usage_summary()}\n"
                                f"- Workspace status: All changes, files, and command outputs up to this point are preserved on disk.\n\n"
                                f"*Tip: Add another API key to API_KEYS_OPEN_ROUTER.txt or switch to local models (--local), then type 'Continue' to resume.*"
                            )
                            self._emit(AgentEventType.ERROR, {
                                "error": err_msg,
                                "suggestions": get_model_suggestions(self.model),
                            })
                            return err_msg

                    err_msg = f"API Connection error with model '{self.model}': {e}"
                    self._emit(AgentEventType.ERROR, {
                        "error": err_msg,
                        "suggestions": get_model_suggestions(self.model),
                    })
                    return err_msg

            if not chat.choices:
                err_msg = f"Error: Received empty response from model '{self.model}'."
                self._emit(AgentEventType.ERROR, {
                    "error": err_msg,
                    "suggestions": get_model_suggestions(self.model),
                })
                return err_msg

            # Record token usage
            if hasattr(chat, "usage") and chat.usage:
                self.memory.record_usage(
                    prompt_tokens=getattr(chat.usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(chat.usage, "completion_tokens", 0) or 0,
                )

            choice = chat.choices[0]
            message = choice.message
            tool_calls = message.tool_calls
            content = message.content or ""

            # Check for native tool execution requests
            if tool_calls:
                self.memory.add_assistant_message(
                    content=content,
                    tool_calls=tool_calls,
                )
                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = tool_call.function.arguments

                    # Resolve alias if any
                    resolved = self.tools.get_tool(fn_name)
                    if resolved:
                        fn_name = resolved.name

                    self._emit(AgentEventType.TOOL_CALL, {
                        "id": tool_call.id,
                        "name": fn_name,
                        "arguments": fn_args,
                    })

                    # Dispatch through tool registry
                    result = self.tools.dispatch(fn_name, fn_args)

                    self._emit(AgentEventType.TOOL_RESULT, {
                        "id": tool_call.id,
                        "name": fn_name,
                        "result": result,
                    })

                    self.memory.add_tool_result(tool_call.id, result)
            else:
                # Local models / smaller models often output tool calls directly in text content
                available_tools = set(self.tools._tools.keys())
                action_type, extracted, args_str = extract_tool_or_answer(content, available_tools)

                if action_type == "tool_call":
                    synthetic_id = f"call_{uuid.uuid4().hex[:8]}"
                    synthetic_tool_call = {
                        "id": synthetic_id,
                        "type": "function",
                        "function": {
                            "name": extracted,
                            "arguments": args_str,
                        },
                    }
                    self.memory.add_assistant_message(
                        content=None,
                        tool_calls=[synthetic_tool_call],
                    )

                    self._emit(AgentEventType.TOOL_CALL, {
                        "id": synthetic_id,
                        "name": extracted,
                        "arguments": args_str,
                    })

                    result = self.tools.dispatch(extracted, args_str)

                    self._emit(AgentEventType.TOOL_RESULT, {
                        "id": synthetic_id,
                        "name": extracted,
                        "result": result,
                    })

                    self.memory.add_tool_result(synthetic_id, result)
                else:
                    final_answer = extracted
                    self.memory.add_assistant_message(
                        content=final_answer,
                        tool_calls=None,
                    )
                    self._emit(AgentEventType.ANSWER, final_answer)
                    return final_answer
