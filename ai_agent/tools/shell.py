"""
Sandboxed shell command execution with execution timeouts and safety bounds.
"""

import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional
from ..config import DEFAULT_SHELL_TIMEOUT
from .base import BaseTool


def normalize_python_command(command: str) -> str:
    """
    Replace calls to python/python3/py (and pip/pip3) with the active sys.executable,
    respecting command start or chaining operators (&&, ||, ;, |).
    The Windows launcher's version flag (py -3, py -3.11) is dropped since python.exe rejects it.
    """
    python_exe = f'"{sys.executable}"'
    command = re.sub(
        r'(^|[;&|]\s*)pip3?(?=\s|$)',
        lambda m: f'{m.group(1)}{python_exe} -m pip',
        command,
    )
    pattern = r'(^|[;&|]\s*)(?:python3?|py(?:\s+-\d+(?:\.\d+)?(?:-\d+)?)?)(?=\s|$)'
    return re.sub(pattern, lambda m: f'{m.group(1)}{python_exe}', command)


class BashTool(BaseTool):
    """Execute shell commands strictly anchored in the workspace directory."""

    name = "Bash"
    description = (
        "Execute a shell command within the workspace sandbox. "
        "Commands are executed using the local system shell with a strict timeout."
    )
    parameters = {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {
                "type": "string",
                "description": "The command string to execute in the terminal.",
            }
        },
    }

    def __init__(
        self,
        workspace_dir: Path,
        timeout: int = DEFAULT_SHELL_TIMEOUT,
        approve: Optional[Callable[[str], bool]] = None,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.timeout = timeout
        self.approve = approve

    def execute(self, **kwargs) -> str:
        command = kwargs.get("command") or kwargs.get("cmd")
        if not command and kwargs:
            # Fallback if argument was passed as a single unnamed value
            command = list(kwargs.values())[0]

        if not command or not isinstance(command, str):
            return "Error: No command provided to execute."

        command = command.strip()
        if self.approve and not self.approve(command):
            return "Command rejected by user. Do not retry it; ask the user or try a different approach."
        command = normalize_python_command(command)

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workspace_dir),
                errors="replace",
                start_new_session=sys.platform != "win32",
            )
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                # Kill the whole process tree: killing only the shell leaves children holding the pipes.
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        capture_output=True,
                    )
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            stdout = stdout.strip()
            stderr = stderr.strip()

            result_parts = []
            if process.returncode != 0:
                result_parts.append(f"[Exit code: {process.returncode}]")
            if stdout:
                result_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                result_parts.append(f"STDERR:\n{stderr}")
            if not stdout and not stderr:
                result_parts.append("Command completed with no output.")

            return "\n".join(result_parts)
        except subprocess.TimeoutExpired:
            return (
                f"Error: Command timed out after {self.timeout} seconds and was terminated. "
                "Ensure commands do not start interactive prompts or long-running daemons."
            )
        except Exception as e:
            return f"Error executing command: {type(e).__name__}: {e}"
