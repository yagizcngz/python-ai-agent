"""
Conversation memory and context management for the AI Agent.
"""

import platform
from typing import Any, Dict, List, Optional

from ..config import MAX_CONTEXT_CHARS

KEEP_RECENT_MESSAGES = 6
TRIMMED_TOOL_OUTPUT = "[Old tool output trimmed to save context. Re-run the tool if you need it again.]"


def build_default_system_prompt() -> str:
    """Dynamically build system instructions aware of current host OS and available tools."""
    os_name = platform.system()
    if os_name == "Windows":
        os_guidance = (
            "The host machine runs Windows. When using the Bash tool, use Windows-compatible commands "
            "(e.g., use 'dir' instead of 'ls', and 'mkdir' without '-p')."
        )
    elif os_name == "Darwin":
        os_guidance = "The host machine runs macOS (Darwin). Standard Unix/POSIX commands are supported."
    else:
        os_guidance = "The host machine runs Linux. Standard Unix/POSIX commands are supported."

    return (
        "You are an autonomous AI programming and task assistant.\n"
        f"{os_guidance}\n"
        "Guidelines:\n"
        "1. Prefer native tools (Read, Write, Edit, ListDir) over shell commands whenever possible for file operations.\n"
        "2. When modifying existing files, prefer the Edit tool to change specific blocks rather than rewriting entire files.\n"
        "3. Provide clean, concise answers and confirm when tasks have been completed.\n"
        "4. Work iteratively: inspect files first, plan changes, apply them, and verify results.\n"
        "5. Do NOT use any emojis in your responses or outputs. Keep all text plain, clear, and professional.\n"
        "6. Your current working directory is already the project workspace root. All relative paths are evaluated directly inside this root. "
        "If the user refers to the project root folder name in their instructions, operate directly within the workspace root—do NOT create a nested folder with the same name.\n"
        "7. For conversational questions (such as greetings, introductions, general questions, or questions about yourself), reply directly in natural language plain text. Do NOT call tools or output tool JSON for conversational questions.\n"
        "8. Only call tools when you genuinely need to inspect, modify, or execute files or commands in the workspace. Never invent or hallucinate tool names; only use: Read, Write, Edit, ListDir, Bash."
    )


class ConversationMemory:
    """Manages chat messages history and token metrics."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or build_default_system_prompt()
        self.messages: List[Dict[str, Any]] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.reset()

    def reset(self, new_system_prompt: Optional[str] = None) -> None:
        """Reset conversation history while retaining or updating system prompt."""
        if new_system_prompt:
            self.system_prompt = new_system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Any]] = None,
    ) -> None:
        """Add assistant response or tool call request to history."""
        msg: Dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            formatted_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                    fn_args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                else:
                    tc_id = getattr(tc, "id", "")
                    fn = getattr(tc, "function", None)
                    fn_name = getattr(fn, "name", "") if fn else ""
                    fn_args = getattr(fn, "arguments", "") if fn else ""

                formatted_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    },
                })
            msg["tool_calls"] = formatted_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool execution result to history."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record token usage metrics from an API response."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return conversation history, trimming old tool outputs if it exceeds the context budget."""
        self._trim_old_tool_outputs()
        return list(self.messages)

    def _trim_old_tool_outputs(self) -> None:
        # Shrinks content only; never drops messages, so tool_call/tool_result pairs stay valid.
        total = sum(len(str(m.get("content") or "")) for m in self.messages)
        for msg in self.messages[:-KEEP_RECENT_MESSAGES]:
            if total <= MAX_CONTEXT_CHARS:
                return
            if msg.get("role") == "tool" and msg.get("content") != TRIMMED_TOOL_OUTPUT:
                total -= len(str(msg["content"])) - len(TRIMMED_TOOL_OUTPUT)
                msg["content"] = TRIMMED_TOOL_OUTPUT
