from __future__ import annotations

import asyncio

from pydantic import BaseModel

from batch_agent import BatchAgent
from batch_agent.compiler import TaskCompiler
from batch_agent.spec import AgentError, AgentResult


class ResearchPlan(BaseModel):
    items: list[str]


class ResearchAnswer(BaseModel):
    question: str
    answer: str


class Synthesis(BaseModel):
    count: int
    errors: int = 0


def test_planner_items_dispatch_map_agents(monkeypatch) -> None:
    calls: list[dict] = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        if kwargs["output_schema"] is ResearchPlan:
            return [AgentResult(job_id="job-0", index=0, output=ResearchPlan(items=[f"q{i}" for i in range(5)]))]
        if kwargs["output_schema"] is ResearchAnswer:
            return [
                AgentResult(job_id=f"job-{i}", index=i, output=ResearchAnswer(question=item["item"], answer="ok"))
                for i, item in enumerate(kwargs["inputs"])
            ]
        return [AgentResult(job_id="job-0", index=0, output=Synthesis(count=5))]

    monkeypatch.setattr(BatchAgent, "run", fake_run)

    results, summary = asyncio.run(BatchAgent.run_with_map_reduce(
        plan_prompt="plan",
        plan_output_schema=ResearchPlan,
        task="answer {item}",
        output_schema=ResearchAnswer,
        reduce="reduce {n}",
        reduce_schema=Synthesis,
        model="mock",
        backend="anthropic://",
    ))

    assert len(results) == 5
    assert summary.count == 5
    assert calls[1]["inputs"] == [{"item": f"q{i}", "index": i} for i in range(5)]


def test_reduce_receives_all_map_results(monkeypatch) -> None:
    reduce_inputs = []

    async def fake_run(**kwargs):
        if kwargs["output_schema"] is ResearchPlan:
            return [AgentResult(job_id="job-0", index=0, output=ResearchPlan(items=["a", "b", "c", "d", "e"]))]
        if kwargs["output_schema"] is ResearchAnswer:
            return [
                AgentResult(job_id=f"job-{i}", index=i, output=ResearchAnswer(question=str(i), answer="ok"))
                for i in range(5)
            ]
        reduce_inputs.append(kwargs["inputs"][0]["reduce_prompt"])
        return [AgentResult(job_id="job-0", index=0, output=Synthesis(count=5))]

    monkeypatch.setattr(BatchAgent, "run", fake_run)

    asyncio.run(BatchAgent.run_with_map_reduce(
        plan_prompt="plan",
        plan_output_schema=ResearchPlan,
        task="answer {item}",
        output_schema=ResearchAnswer,
        reduce="reduce {n}",
        reduce_schema=Synthesis,
        model="mock",
        backend="anthropic://",
    ))

    assert reduce_inputs
    assert reduce_inputs[0].count('"status": "ok"') == 5
    assert '"question": "0"' in reduce_inputs[0]


def test_partial_map_failure_is_passed_to_reduce(monkeypatch) -> None:
    reduce_inputs = []

    async def fake_run(**kwargs):
        if kwargs["output_schema"] is ResearchPlan:
            return [AgentResult(job_id="job-0", index=0, output=ResearchPlan(items=[f"q{i}" for i in range(5)]))]
        if kwargs["output_schema"] is ResearchAnswer:
            return [
                AgentResult(job_id="job-0", index=0, output=ResearchAnswer(question="q0", answer="ok")),
                AgentResult(job_id="job-1", index=1, output=None, error=AgentError("ToolError", "failed")),
                AgentResult(job_id="job-2", index=2, output=ResearchAnswer(question="q2", answer="ok")),
                AgentResult(job_id="job-3", index=3, output=ResearchAnswer(question="q3", answer="ok")),
                AgentResult(job_id="job-4", index=4, output=ResearchAnswer(question="q4", answer="ok")),
            ]
        reduce_inputs.append(kwargs["inputs"][0]["reduce_prompt"])
        return [AgentResult(job_id="job-0", index=0, output=Synthesis(count=5, errors=1))]

    monkeypatch.setattr(BatchAgent, "run", fake_run)

    _, summary = asyncio.run(BatchAgent.run_with_map_reduce(
        plan_prompt="plan",
        plan_output_schema=ResearchPlan,
        task="answer {item}",
        output_schema=ResearchAnswer,
        reduce="reduce {n}",
        reduce_schema=Synthesis,
        model="mock",
        backend="anthropic://",
    ))

    assert summary.errors == 1
    assert reduce_inputs[0].count('"status": "ok"') == 4
    assert reduce_inputs[0].count('"status": "error"') == 1


def test_checkpoint_dir_is_split_by_stage(monkeypatch, tmp_path) -> None:
    checkpoint_dirs = []

    async def fake_run(**kwargs):
        checkpoint_dirs.append(kwargs.get("checkpoint_dir"))
        if kwargs["output_schema"] is ResearchPlan:
            return [AgentResult(job_id="job-0", index=0, output=ResearchPlan(items=["q"]))]
        if kwargs["output_schema"] is ResearchAnswer:
            return [AgentResult(job_id="job-0", index=0, output=ResearchAnswer(question="q", answer="ok"))]
        return [AgentResult(job_id="job-0", index=0, output=Synthesis(count=1))]

    monkeypatch.setattr(BatchAgent, "run", fake_run)

    asyncio.run(BatchAgent.run_with_map_reduce(
        plan_prompt="plan",
        plan_output_schema=ResearchPlan,
        task="answer {item}",
        output_schema=ResearchAnswer,
        reduce="reduce {n}",
        reduce_schema=Synthesis,
        model="mock",
        backend="anthropic://",
        checkpoint_dir=str(tmp_path),
    ))

    assert checkpoint_dirs == [
        str(tmp_path / "plan"),
        str(tmp_path / "map"),
        str(tmp_path / "reduce"),
    ]


def test_task_compiler_builds_three_tier_dag_descriptor() -> None:
    dag = TaskCompiler().build_dag("map_reduce")

    assert dag["stages"] == ["plan", "map", "reduce"]
    assert dag["edges"] == [("plan", "map"), ("map", "reduce")]
