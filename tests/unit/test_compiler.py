from __future__ import annotations

import pytest
from pydantic import BaseModel

from batch_agent.compiler import SCHEMA_INSTRUCTION, TaskCompiler
from batch_agent.spec import BatchSpec


class Summary(BaseModel):
    title: str


def test_compiler_injects_schema_and_builds_jobs() -> None:
    spec = BatchSpec(task="Summarize {text}", inputs=[{"text": "alpha"}], output_schema=Summary)
    plan = TaskCompiler().compile(spec)

    assert "valid JSON object" in plan.shared.prefix
    assert plan.shared.schema is not None
    assert plan.jobs[0].prompt == "Summarize alpha"


def test_compiler_reports_missing_template_variable() -> None:
    spec = BatchSpec(task="Summarize {text}", inputs=[{"body": "alpha"}])

    with pytest.raises(ValueError, match="missing template variable"):
        TaskCompiler().compile(spec)


def test_compiler_flags_oversized_jobs() -> None:
    spec = BatchSpec(task="Summarize {text}", inputs=[{"text": "x" * 100}], model_max_context=10, min_response_tokens=1)
    plan = TaskCompiler().compile(spec)

    assert plan.jobs[0].oversized is True


def test_compiler_hoists_constant_inputs() -> None:
    spec = BatchSpec(task="You are in {domain}. Summarize {text}", inputs=[{"domain": "biology", "text": "a"}, {"domain": "biology", "text": "b"}])
    plan = TaskCompiler().compile(spec)

    assert plan.shared.hoisted_inputs == {"domain": "biology"}
    assert "biology" in plan.shared.prefix


def test_compiler_does_not_duplicate_schema_instruction() -> None:
    spec = BatchSpec(
        task="Summarize {text}",
        inputs=[{"text": "alpha"}],
        system_prompt=f"Custom instructions.\n\n{SCHEMA_INSTRUCTION}\nSchema:\n{{}}",
        output_schema=Summary,
    )

    plan = TaskCompiler().compile(spec)

    assert plan.shared.schema is not None
    assert plan.shared.prefix.count(SCHEMA_INSTRUCTION) == 1
