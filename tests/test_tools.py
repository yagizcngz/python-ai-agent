"""
Test suite for filesystem tools, shell execution, timeouts, and argument parsing.
"""

from pathlib import Path
import sys

from ai_agent.tools.base import ToolRegistry
from ai_agent.tools.filesystem import EditTool, ListDirTool, ReadTool, WriteTool
from ai_agent.tools.shell import BashTool, normalize_python_command


def test_write_and_read_tools(tmp_path: Path):
    write_tool = WriteTool(tmp_path)
    read_tool = ReadTool(tmp_path)

    # Test writing nested file
    res_write = write_tool.execute(file_path="src/app.py", content="print('hello world')")
    assert "written successfully" in res_write

    # Test reading written file
    res_read = read_tool.execute(file_path="src/app.py")
    assert res_read == "print('hello world')"


def test_read_nonexistent_file(tmp_path: Path):
    read_tool = ReadTool(tmp_path)
    res = read_tool.execute(file_path="missing.txt")
    assert "does not exist" in res


def test_edit_tool(tmp_path: Path):
    write_tool = WriteTool(tmp_path)
    edit_tool = EditTool(tmp_path)
    read_tool = ReadTool(tmp_path)

    # Initial file
    write_tool.execute(file_path="code.py", content="def add(a, b):\n    return a - b\n")

    # Perform edit
    res_edit = edit_tool.execute(
        file_path="code.py",
        old_content="return a - b",
        new_content="return a + b",
    )
    assert "edited successfully" in res_edit

    # Verify content
    updated = read_tool.execute(file_path="code.py")
    assert "return a + b" in updated


def test_edit_tool_not_found(tmp_path: Path):
    write_tool = WriteTool(tmp_path)
    edit_tool = EditTool(tmp_path)

    write_tool.execute(file_path="code.py", content="x = 10")
    res = edit_tool.execute(file_path="code.py", old_content="y = 20", new_content="y = 30")
    assert "Target text not found" in res


def test_edit_tool_ambiguous(tmp_path: Path):
    write_tool = WriteTool(tmp_path)
    edit_tool = EditTool(tmp_path)

    write_tool.execute(file_path="code.py", content="x = 1\nx = 1\n")
    res = edit_tool.execute(file_path="code.py", old_content="x = 1", new_content="x = 2")
    assert "Ambiguous edit" in res


def test_listdir_tool(tmp_path: Path):
    write_tool = WriteTool(tmp_path)
    listdir_tool = ListDirTool(tmp_path)

    write_tool.execute(file_path="file1.txt", content="a")
    write_tool.execute(file_path="sub/file2.txt", content="b")

    res = listdir_tool.execute(directory_path=".")
    assert "file1.txt" in res
    assert "[DIR] sub" in res


def test_bash_tool_echo(tmp_path: Path):
    bash_tool = BashTool(tmp_path)
    res = bash_tool.execute(command='echo "testing bash"')
    assert "testing bash" in res


def test_bash_tool_timeout(tmp_path: Path):
    # Set short timeout of 1 second and run sleep
    bash_tool = BashTool(tmp_path, timeout=1)
    res = bash_tool.execute(command='python -c "import time; time.sleep(5)"')
    assert "timed out after 1 seconds" in res


def test_normalize_python_command():
    res = normalize_python_command("python test.py")
    assert "python" not in res[:7] or '"' in res

    res_chained = normalize_python_command("echo 1 && python3 script.py")
    assert "script.py" in res_chained


def test_tool_registry_malformed_json():
    registry = ToolRegistry()
    # Malformed JSON should return an error message rather than crashing
    res = registry.dispatch("nonexistent", "{invalid json")
    assert "not found" in res

    # Register dummy tool and test malformed json
    class DummyTool(WriteTool):
        pass

    dummy = DummyTool(Path("."))
    registry.register(dummy)
    res_bad_json = registry.dispatch(dummy.name, "not json at all")
    assert "Malformed JSON arguments" in res_bad_json


def test_bash_timeout_kills_child_processes(tmp_path: Path):
    import time
    # The shell's child holds the output pipe; the timeout must kill the whole tree, not wait for it.
    bash_tool = BashTool(tmp_path, timeout=1)
    start = time.time()
    res = bash_tool.execute(command='python -c "import time; time.sleep(20)"')
    assert "timed out" in res
    assert time.time() - start < 8


def test_bash_tool_rejected_command_does_not_run(tmp_path: Path):
    seen = []
    bash_tool = BashTool(tmp_path, approve=lambda cmd: seen.append(cmd) or False)
    res = bash_tool.execute(command='echo hi > made.txt')
    assert "rejected" in res
    assert seen == ["echo hi > made.txt"]
    assert not (tmp_path / "made.txt").exists()


def test_write_empty_file(tmp_path: Path):
    res = WriteTool(tmp_path).execute(file_path="empty.txt", content="")
    assert "written successfully" in res
    assert (tmp_path / "empty.txt").read_bytes() == b""


def test_edit_keeps_line_endings(tmp_path: Path):
    (tmp_path / "lf.py").write_bytes(b"a = 1\nb = 2\n")
    (tmp_path / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    edit_tool = EditTool(tmp_path)
    edit_tool.execute(file_path="lf.py", old_content="b = 2", new_content="b = 3")
    edit_tool.execute(file_path="crlf.py", old_content="b = 2", new_content="b = 3")
    assert (tmp_path / "lf.py").read_bytes() == b"a = 1\nb = 3\n"
    assert (tmp_path / "crlf.py").read_bytes() == b"a = 1\r\nb = 3\r\n"


def test_edit_multiline_on_crlf_file(tmp_path: Path):
    (tmp_path / "w.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    # The model copies text from Read output, which uses LF only
    assert "\r" not in ReadTool(tmp_path).execute(file_path="w.py")
    res = EditTool(tmp_path).execute(file_path="w.py", old_content="a = 1\nb = 2", new_content="a = 1\nb = 20")
    assert "edited successfully" in res
    assert (tmp_path / "w.py").read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"


def test_edit_refuses_non_utf8(tmp_path: Path):
    (tmp_path / "latin.txt").write_bytes("café = 1\n".encode("latin-1"))
    res = EditTool(tmp_path).execute(file_path="latin.txt", old_content="= 1", new_content="= 2")
    assert "not valid UTF-8" in res
    assert (tmp_path / "latin.txt").read_bytes() == "café = 1\n".encode("latin-1")


def test_read_line_range_and_binary(tmp_path: Path):
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
    read_tool = ReadTool(tmp_path)
    res = read_tool.execute(file_path="f.txt", offset=3, limit=2)
    assert res == "[Lines 3-4 of 10]\nline3\nline4\n"
    assert "past the end" in read_tool.execute(file_path="f.txt", offset=50)
    assert "binary" in read_tool.execute(file_path="b.bin")


def test_normalize_python_command_launcher_and_pip():
    exe = f'"{sys.executable}"'
    assert normalize_python_command("py -3 x.py") == f"{exe} x.py"
    assert normalize_python_command("cd a && py -3.11 -m pytest") == f"cd a && {exe} -m pytest"
    assert normalize_python_command("pip install rich") == f"{exe} -m pip install rich"
    assert normalize_python_command("mypy x.py") == "mypy x.py"
    assert normalize_python_command("pipenv install") == "pipenv install"
