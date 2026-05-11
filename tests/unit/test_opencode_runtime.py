from __future__ import annotations

import asyncio
import json

from batch_agent import BatchAgent
from batch_agent.runtimes import OpenCodeRuntime
from batch_agent.runtimes import opencode
from batch_agent.runtimes.opencode import parse_opencode_jsonl


class FakeProc:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True


def test_build_config_content_for_sglang() -> None:
    runtime = OpenCodeRuntime(
        backend="sglang://localhost:30000",
        model="Qwen/Qwen2.5-32B-Instruct",
    )

    config = json.loads(runtime.build_config_content())

    model = config["provider"]["sglang"]["models"]["qwen32b"]
    assert model["id"] == "Qwen/Qwen2.5-32B-Instruct"
    assert model["tool_call"] is True
    assert config["provider"]["sglang"]["options"]["baseURL"] == "http://localhost:30000/v1"


def test_parse_opencode_jsonl_extracts_text_events() -> None:
    stdout = "\n".join([
        json.dumps({"type": "start"}),
        json.dumps({"type": "text", "text": "hello "}),
        json.dumps({"type": "text", "content": "world"}),
        json.dumps({"type": "done"}),
    ])

    events, text = parse_opencode_jsonl(stdout)

    assert len(events) == 4
    assert text == "hello world"


def test_generate_runs_opencode_with_config_env(monkeypatch) -> None:
    seen = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return FakeProc(stdout=b'{"type":"text","text":"review ok"}\n')

    monkeypatch.setattr(opencode.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    runtime = OpenCodeRuntime(
        backend="sglang://localhost:30000",
        model="Qwen/Qwen2.5-32B-Instruct",
        working_dir="/tmp/project",
    )

    response = asyncio.run(
        runtime.generate(
            shared=opencode.SharedContext(prefix="shared system"),
            job=opencode.AgentJob(
                job_id="job-0",
                index=0,
                input_data={},
                prompt="Review file.py",
                estimated_prompt_tokens=1,
            ),
            model=runtime.model,
            timeout=5,
        )
    )

    assert response.content == "review ok"
    assert seen["args"][:7] == (
        "opencode",
        "run",
        "--model",
        "sglang/qwen32b",
        "--format",
        "json",
        "--dangerously-skip-permissions",
    )
    assert seen["args"][7] == "shared system\n\nReview file.py"
    assert seen["kwargs"]["cwd"] == "/tmp/project"
    env_config = json.loads(seen["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert env_config["provider"]["sglang"]["options"]["baseURL"] == "http://localhost:30000/v1"


def test_batchagent_accepts_runtime_and_max_agents(monkeypatch) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc(stdout=b'{"type":"text","text":"done"}\n')

    async def fake_warm_prefix(self, shared, model):
        return "warm-key"

    monkeypatch.setattr(opencode.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(OpenCodeRuntime, "warm_prefix", fake_warm_prefix)

    results = asyncio.run(
        BatchAgent.run(
            runtime=OpenCodeRuntime(
                backend="sglang://localhost:30000",
                model="Qwen/Qwen2.5-32B-Instruct",
            ),
            task="Review {file}",
            inputs=[{"file": "a.py"}, {"file": "b.py"}],
            max_agents=2,
            streaming_tool_dispatch=False,
        )
    )

    assert [result.output for result in results] == ["done", "done"]
