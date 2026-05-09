from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel

import batch_agent
from batch_agent import BatchAgent
from batch_agent.backends import BackendAdapter, BackendResponse
from batch_agent.spec import AgentJob, Message, SharedContext


class ResearchPlan(BaseModel):
    items: list[str]


class ResearchAnswer(BaseModel):
    question: str
    answer: str


class Survey(BaseModel):
    count: int
    questions: list[str]


class MapReduceMockBackend(BackendAdapter):
    def __init__(self) -> None:
        self.stage_order: list[str] = []

    async def generate(self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None, model: str, tools=None, timeout: float | None = None) -> BackendResponse:
        prompt = messages[-1].content if messages else job.prompt
        if "Generate plan" in prompt:
            self.stage_order.append("plan")
            return BackendResponse(content=json.dumps({"items": ["alpha", "beta", "gamma"]}), stop_reason="end_turn")
        if "Results:" in prompt:
            self.stage_order.append("reduce")
            questions = []
            payload = json.loads(prompt[prompt.index("Results:") + len("Results:"):])
            for item in payload:
                if item["status"] == "ok":
                    questions.append(item["output"]["question"])
            return BackendResponse(content=json.dumps({"count": len(payload), "questions": questions}), stop_reason="end_turn")

        self.stage_order.append("map")
        question = prompt.rsplit(" ", 1)[-1]
        return BackendResponse(
            content=json.dumps({"question": question, "answer": f"answer {question}"}),
            stop_reason="end_turn",
        )


def test_three_tier_map_reduce_e2e_against_mock_backend(monkeypatch) -> None:
    backend = MapReduceMockBackend()
    original_from_url = batch_agent.backend_from_url
    monkeypatch.setattr(batch_agent, "backend_from_url", lambda url: backend if url == "mock://" else original_from_url(url))

    seen = []

    async def run() -> None:
        results, survey = await BatchAgent.run_with_map_reduce(
            plan_prompt="Generate plan",
            plan_output_schema=ResearchPlan,
            task="Research {item}",
            output_schema=ResearchAnswer,
            reduce="Synthesize {n}",
            reduce_schema=Survey,
            model="mock",
            backend="mock://",
            max_concurrent=3,
            on_result=lambda r: seen.append(r.output.question),
        )
        assert [r.output.question for r in results] == ["alpha", "beta", "gamma"]
        assert survey.count == 3
        assert survey.questions == ["alpha", "beta", "gamma"]

    asyncio.run(run())

    assert backend.stage_order[0] == "plan"
    assert backend.stage_order[1:4] == ["map", "map", "map"]
    assert backend.stage_order[-1] == "reduce"
    assert sorted(seen) == ["alpha", "beta", "gamma"]
