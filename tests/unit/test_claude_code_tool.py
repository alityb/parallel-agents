from __future__ import annotations

import asyncio

import pytest

from batch_agent.tools import Tool, ToolError
from batch_agent.tools import builtin


class FakeProc:
    def __init__(self, *, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def test_claude_code_success(monkeypatch) -> None:
    seen = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return FakeProc(returncode=0, stdout=b'{"result": "done"}')

    monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    result = asyncio.run(Tool.claude_code.func("inspect repo", working_dir="/tmp/project"))

    assert result == "done"
    assert seen["args"] == (
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
        "inspect repo",
    )
    assert seen["kwargs"]["cwd"] == "/tmp/project"


def test_claude_code_nonzero_return(monkeypatch) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc(returncode=2, stderr=b"bad auth token and details")

    monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(ToolError, match="claude CLI failed: bad auth token"):
        asyncio.run(Tool.claude_code.func("run task"))


def test_claude_code_not_installed(monkeypatch) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(ToolError, match="npm install -g @anthropic-ai/claude-code"):
        asyncio.run(Tool.claude_code.func("run task"))
