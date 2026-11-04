"""
Sandboxed filesystem tools for Read, Write, Edit, and ListDir.
"""

import os
from pathlib import Path
from typing import List
from ..config import MAX_TOOL_OUTPUT_CHARS
from .base import BaseTool


def resolve_sandbox_path(file_path: str, workspace_dir: Path) -> Path:
    """
    Resolve a relative or absolute path against workspace_dir and enforce
    that the target path remains strictly within the workspace sandbox.
    """
    workspace = workspace_dir.resolve()
    # If file_path is absolute and points inside workspace, that's fine.
    # If it's relative, anchor it to workspace.
    raw_path = Path(file_path)
    if raw_path.is_absolute():
        target = raw_path.resolve()
    else:
        target = (workspace / raw_path).resolve()

    if not target.is_relative_to(workspace):
        raise PermissionError(
            f"Security Error: Access denied. Path '{file_path}' resolves outside the workspace sandbox '{workspace}'."
        )
    return target


class ReadTool(BaseTool):
    """Safely read file contents inside workspace."""

    name = "Read"
    description = (
        "Read and return the contents of a file within the workspace sandbox. "
        "For long files, pass offset/limit to read a range of lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path to the file to read (e.g. 'main.py' or 'src/utils.py').",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from (optional).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return (optional).",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path") or kwargs.get("path") or kwargs.get("filename")
        if not file_path:
            return "Error: Missing required argument 'file_path'."

        try:
            target = resolve_sandbox_path(file_path, self.workspace_dir)
            if not target.exists():
                return f"Error: File '{file_path}' does not exist."
            if target.is_dir():
                return f"Error: '{file_path}' is a directory, not a file. Use ListDir to inspect directories."

            raw = target.read_bytes()
            if b"\x00" in raw[:8192]:
                return f"Error: '{file_path}' looks like a binary file ({len(raw)} bytes); it cannot be shown as text."
            content = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")

            offset = kwargs.get("offset")
            limit = kwargs.get("limit")
            if offset is None and limit is None:
                if len(content) > MAX_TOOL_OUTPUT_CHARS:
                    total = content.count("\n") + 1
                    content += (
                        f"\n\n[Note: file has {total} lines and will be shortened in the middle; "
                        "use offset/limit to read the part you need.]"
                    )
                return content

            lines = content.splitlines(keepends=True)
            start = max(int(offset or 1), 1) - 1
            end = len(lines) if limit is None else start + max(int(limit), 0)
            if start >= len(lines):
                return f"Error: offset {start + 1} is past the end of '{file_path}' ({len(lines)} lines)."
            return f"[Lines {start + 1}-{min(end, len(lines))} of {len(lines)}]\n" + "".join(lines[start:end])
        except PermissionError as pe:
            return str(pe)
        except Exception as e:
            return f"Error reading file '{file_path}': {e}"


class WriteTool(BaseTool):
    """Safely write or overwrite file contents inside workspace."""

    name = "Write"
    description = "Write content to a file inside the workspace sandbox, creating any required parent directories."
    parameters = {
        "type": "object",
        "required": ["file_path", "content"],
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path of the file to write to (e.g. 'src/app.py').",
            },
            "content": {
                "type": "string",
                "description": "The exact text content to write to the file.",
            },
        },
    }

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path") or kwargs.get("path") or kwargs.get("filename")
        content = kwargs["content"] if "content" in kwargs else kwargs.get("file_content")

        if not file_path:
            return "Error: Missing required argument 'file_path'."
        if content is None:
            return "Error: Missing required argument 'content'."

        try:
            target = resolve_sandbox_path(file_path, self.workspace_dir)
            target.parent.mkdir(parents=True, exist_ok=True)

            with open(target, "w", encoding="utf-8", errors="replace", newline="") as f:
                f.write(content)

            return f"File '{file_path}' written successfully ({len(content)} characters)."
        except PermissionError as pe:
            return str(pe)
        except Exception as e:
            return f"Error writing file '{file_path}': {e}"


class EditTool(BaseTool):
    """In-place search-and-replace editing without rewriting the entire file."""

    name = "Edit"
    description = (
        "Perform an in-place edit on an existing file by replacing an exact block of text (old_content) "
        "with new text (new_content). The old_content must uniquely match text in the file."
    )
    parameters = {
        "type": "object",
        "required": ["file_path", "old_content", "new_content"],
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_content": {
                "type": "string",
                "description": "Exact text or lines to find and replace.",
            },
            "new_content": {
                "type": "string",
                "description": "Replacement text.",
            },
        },
    }

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path") or kwargs.get("path") or kwargs.get("filename")
        old_content = kwargs.get("old_content")
        new_content = kwargs.get("new_content")

        if not file_path:
            return "Error: Missing required argument 'file_path'."
        if old_content is None or new_content is None:
            return "Error: Both 'old_content' and 'new_content' are required for editing."

        try:
            target = resolve_sandbox_path(file_path, self.workspace_dir)
            if not target.is_file():
                return f"Error: File '{file_path}' does not exist or is a directory."

            # newline="" keeps the file's own line endings (no LF -> CRLF rewrite on Windows).
            # Strict decoding: writing back replacement characters would corrupt non-UTF-8 files.
            try:
                with open(target, "r", encoding="utf-8", newline="") as f:
                    current_text = f.read()
            except UnicodeDecodeError:
                return (
                    f"Error: '{file_path}' is not valid UTF-8 text; refusing to edit it to avoid corrupting it. "
                    "Use Write to replace the whole file if that is intended."
                )

            # Read shows LF only, so match CRLF files by converting the model's text to CRLF
            if "\r\n" in current_text and "\r\n" not in old_content:
                old_content = old_content.replace("\n", "\r\n")
                new_content = new_content.replace("\n", "\r\n")

            occurrences = current_text.count(old_content)
            if occurrences == 0:
                return (
                    f"Error: Target text not found in '{file_path}'. "
                    "Please inspect the file using Read first to ensure exact matching characters and whitespace."
                )
            if occurrences > 1:
                return (
                    f"Error: Ambiguous edit. Target text appears {occurrences} times in '{file_path}'. "
                    "Include more surrounding context lines in 'old_content' to make it unique."
                )

            updated_text = current_text.replace(old_content, new_content, 1)
            with open(target, "w", encoding="utf-8", errors="replace", newline="") as f:
                f.write(updated_text)

            return f"File '{file_path}' edited successfully."
        except PermissionError as pe:
            return str(pe)
        except Exception as e:
            return f"Error editing file '{file_path}': {e}"


class ListDirTool(BaseTool):
    """List directory contents inside the sandbox without executing shell commands."""

    name = "ListDir"
    description = "List files and directories in the workspace sandbox."
    parameters = {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "Relative directory path to inspect (defaults to root workspace '.').",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list subdirectories recursively (default: false).",
                "default": False,
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth for recursive listing (default: 2).",
                "default": 2,
            },
        },
    }

    IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".idea", ".vscode"}

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()

    def execute(self, **kwargs) -> str:
        dir_path = kwargs.get("directory_path", ".") or "."
        recursive = bool(kwargs.get("recursive", False))
        max_depth = int(kwargs.get("max_depth", 2))

        try:
            target = resolve_sandbox_path(dir_path, self.workspace_dir)
            if not target.exists():
                return f"Error: Directory '{dir_path}' does not exist."
            if not target.is_dir():
                return f"Error: '{dir_path}' is a file, not a directory. Use Read instead."

            lines: List[str] = []
            rel_root = target.relative_to(self.workspace_dir)
            lines.append(f"Directory: {rel_root if str(rel_root) != '.' else './'}")

            if not recursive:
                entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                for entry in entries:
                    if entry.name in self.IGNORE_DIRS:
                        continue
                    prefix = "[DIR] " if entry.is_dir() else "      "
                    size_info = f" ({entry.stat().st_size} bytes)" if entry.is_file() else ""
                    lines.append(f"  {prefix}{entry.name}{size_info}")
            else:
                base_depth = len(target.parts)
                for root, dirs, files in os.walk(target):
                    dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                    curr_depth = len(Path(root).parts) - base_depth
                    if curr_depth > max_depth:
                        dirs[:] = []
                        continue
                    indent = "  " * (curr_depth + 1)
                    rel_p = Path(root).relative_to(target)
                    if str(rel_p) != ".":
                        lines.append(f"{indent}[DIR] {Path(root).name}/")
                    for f in sorted(files):
                        file_indent = "  " * (curr_depth + 1)
                        lines.append(f"{file_indent}      {f}")

            return "\n".join(lines)
        except PermissionError as pe:
            return str(pe)
        except Exception as e:
            return f"Error listing directory '{dir_path}': {e}"
