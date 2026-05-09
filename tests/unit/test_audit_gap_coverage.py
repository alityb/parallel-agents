from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from batch_agent import BatchAgent
from batch_agent.backends import BackendAdapter, BackendResponse, ParsedToolCall
from batch_agent.checkpoint import CheckpointStore
from batch_agent.cli import main as cli_main
from batch_agent.compiler import TaskCompiler
from batch_agent.compaction import compact_messages_async
from batch_agent.priority_semaphore import PrioritySemaphore
from batch_agent.repair import OutputValidationError, parse_and_validate_output
from batch_agent.scheduler import WaveScheduler
from batch_agent.spec import AgentJob, BatchSpec, Message, SharedContext
from batch_agent.state import AgentState, AgentStatus
from batch_agent.tools import Tool
from batch_agent.tools.pool import ToolPool


class Payload(BaseModel):
    value: int


class Summary(BaseModel):
    ok_count: int
    error_count: int


class StaticBackend(BackendAdapter):
    def __init__(self, content: str = '{"value": 1}', fail: bool = False, delay: float = 0.0) -> None:
        self.content = content
        self.fail = fail
        self.delay = delay
        self.calls = 0

    async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout=None) -> BackendResponse:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("backend down")
        return BackendResponse(content=self.content, raw={"content": [{"type": "text", "text": self.content}]}, stop_reason="end_turn")


def test_compaction_triggers_and_shortens_context() -> None:
    long = "x" * 1000
    messages = [
        Message("user", "start"),
        Message("assistant_raw", json.dumps([{"type": "text", "text": "a1"}])),
        Message("tool_result", json.dumps([{"type": "tool_result", "tool_use_id": "t1", "content": long}])),
        Message("assistant_raw", json.dumps([{"type": "text", "text": "a2"}])),
        Message("tool_result", json.dumps([{"type": "tool_result", "tool_use_id": "t2", "content": long}])),
        Message("assistant_raw", json.dumps([{"type": "text", "text": "a3"}])),
        Message("tool_result", json.dumps([{"type": "tool_result", "tool_use_id": "t3", "content": long}])),
    ]
    before = sum(len(m.content) for m in messages)
    compacted = asyncio.run(compact_messages_async(messages, current_turn=3))
    after = sum(len(m.content) for m in compacted)
    assert after < before
    assert "COMPACTED" in "".join(m.content for m in compacted)


def test_checkpoint_resume_loads_in_progress_state() -> None:
    class ResumeBackend(BackendAdapter):
        def __init__(self) -> None:
            self.seen_resumed_messages = False

        async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout=None) -> BackendResponse:
            self.seen_resumed_messages = bool(messages and len(messages) > 1)
            return BackendResponse(content='{"value": 7}', raw={"content": [{"type": "text", "text": '{"value": 7}'}]}, stop_reason="end_turn")

    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(tmp)
        store.save_state(AgentState(
            job_id="job-0",
            status=AgentStatus.RUNNING,
            turn=1,
            messages=[Message("user", "Do 0"), Message("tool_result", "[]")],
        ))
        store.close()
        spec = BatchSpec(task="Do {x}", inputs=[{"x": 0}], output_schema=Payload, checkpoint_dir=tmp, max_turns=3)
        backend = ResumeBackend()
        results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), backend).run())
        assert results[0].ok
        assert results[0].output.value == 7
        assert backend.seen_resumed_messages is True


def test_batch_collector_batches_30_calls() -> None:
    batch_calls: list[list[int]] = []

    @Tool.batchable(key_arg="item_id", batch_query="SELECT * FROM items WHERE id IN ({ids})")
    async def get_item(item_id: int) -> str:
        return f"single-{item_id}"

    async def batch_handler(ids: list[int]) -> list[str]:
        batch_calls.append(ids)
        return [f"batch-{i}" for i in ids]

    setattr(get_item, "_batch_handler", batch_handler)
    Tool.define(get_item, name="batch_gap_get_item")

    async def run() -> list[str]:
        pool = ToolPool()
        return await asyncio.gather(*[
            pool.call("batch_gap_get_item", {"item_id": i}) for i in range(30)
        ])

    results = asyncio.run(run())
    assert results == [f"batch-{i}" for i in range(30)]
    assert len(batch_calls) == 1
    assert sorted(batch_calls[0]) == list(range(30))


def test_reduce_receives_partial_failures() -> None:
    class ReduceBackend(BackendAdapter):
        async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout=None) -> BackendResponse:
            prompt = messages[-1].content if messages else job.prompt
            if "Results:" in prompt:
                data = json.loads(prompt[prompt.index("["):])
                ok = sum(1 for item in data if item["status"] == "ok")
                err = sum(1 for item in data if item["status"] == "error")
                body = json.dumps({"ok_count": ok, "error_count": err})
                return BackendResponse(content=body, raw={"content": [{"type": "text", "text": body}]}, stop_reason="end_turn")
            if job.index < 5:
                raise RuntimeError("planned map failure")
            body = json.dumps({"value": job.index})
            return BackendResponse(content=body, raw={"content": [{"type": "text", "text": body}]}, stop_reason="end_turn")

    with patch("batch_agent.backend_from_url", return_value=ReduceBackend()):
        results, summary = asyncio.run(BatchAgent.run_with_reduce(
            task="Map {i}",
            inputs=[{"i": i} for i in range(20)],
            output_schema=Payload,
            backend="anthropic://",
            model="mock",
            max_retries=0,
            reduce="Count statuses for {n} results.",
            reduce_schema=Summary,
        ))
    assert sum(1 for r in results if r.ok) == 15
    assert sum(1 for r in results if not r.ok) == 5
    assert summary.ok_count == 15
    assert summary.error_count == 5


def test_retry_exhaustion_returns_structured_error() -> None:
    spec = BatchSpec(task="Do {x}", inputs=[{"x": 1}], output_schema=Payload, max_retries=3)
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), StaticBackend(fail=True)).run())
    assert results[0].ok is False
    assert results[0].attempts == 4
    assert results[0].error is not None
    assert results[0].error.type == "RuntimeError"


def test_timeout_per_turn_retries_then_fails_cleanly() -> None:
    spec = BatchSpec(task="Do {x}", inputs=[{"x": 1}], output_schema=Payload, max_retries=1, timeout_per_turn=0.01)
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), StaticBackend(delay=0.05)).run())
    assert results[0].ok is False
    assert results[0].attempts == 2
    assert results[0].error is not None
    assert results[0].error.type == "TimeoutError"


def test_repair_missing_json_and_schema_failures() -> None:
    with pytest.raises(OutputValidationError):
        parse_and_validate_output("plain prose", Payload)
    assert parse_and_validate_output('{"value": {"nested": 1,},}', dict) == {"value": {"nested": 1}}
    with pytest.raises(ValidationError):
        parse_and_validate_output('{"wrong": 1}', Payload)


def test_cli_smoke_dispatches_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"task": "Do {x}", "inputs": [{"x": 1}], "backend": "mock://"}), encoding="utf-8")

    async def fake_run(**kwargs):
        return [type("R", (), {"job_id": "job-0", "index": 0, "ok": True, "output": {"x": 1}, "error": None, "attempts": 1})()]

    with patch("batch_agent.cli.BatchAgent.run", side_effect=fake_run):
        cli_main(["run", "--spec", str(spec_path)])
    out = capsys.readouterr().out
    assert '"job_id": "job-0"' in out


def test_async_on_result_fires_per_result() -> None:
    seen: list[int] = []

    async def callback(result):
        await asyncio.sleep(0)
        seen.append(result.index)

    spec = BatchSpec(task="Do {x}", inputs=[{"x": 1}, {"x": 2}], output_schema=Payload, on_result=callback)
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), StaticBackend()).run())
    assert [r.index for r in results] == [0, 1]
    assert sorted(seen) == [0, 1]


def test_dynamic_priority_near_done_agent_jumps_ahead() -> None:
    async def run() -> list[str]:
        sem = PrioritySemaphore(1)
        order: list[str] = []
        await sem.acquire(priority=0)

        async def waiter(name: str, priority: float) -> None:
            await sem.acquire(priority=priority)
            order.append(name)
            sem.release()

        fresh = asyncio.create_task(waiter("fresh-turn-1-of-5", priority=4))
        near_done = asyncio.create_task(waiter("near-done-turn-3-of-5", priority=1))
        await asyncio.sleep(0)
        sem.release()
        await asyncio.gather(fresh, near_done)
        return order

    assert asyncio.run(run())[0] == "near-done-turn-3-of-5"


def test_openai_multiturn_end_to_end_scheduler() -> None:
    class OpenAIStyleBackend(BackendAdapter):
        async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout=None) -> BackendResponse:
            has_tool_result = any(m.role == "tool_result" for m in (messages or []))
            if not has_tool_result:
                return BackendResponse(
                    content="",
                    raw={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "gap_echo", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]},
                    tool_calls=[ParsedToolCall(id="call_1", name="gap_echo", args={})],
                    stop_reason="tool_use",
                )
            return BackendResponse(content='{"value": 9}', raw={"choices": [{"message": {"content": '{"value": 9}'}, "finish_reason": "stop"}]}, stop_reason="stop")

    @Tool.define(name="gap_echo", cacheable=False)
    async def gap_echo() -> str:
        return "ok"

    spec = BatchSpec(task="Use tool", inputs=[{}], tools=[Tool.registry["gap_echo"]], output_schema=Payload, max_turns=3)
    results = asyncio.run(WaveScheduler(TaskCompiler().compile(spec), OpenAIStyleBackend()).run())
    assert results[0].ok
    assert results[0].output.value == 9
