from __future__ import annotations

from tests.benchmarks.bench_opencode_baseline import (
    TaskResult,
    output_equivalence_audit,
    result_fingerprints,
)


def make_result(task_id: str, found_bug_ids: list[str], output: str) -> TaskResult:
    fingerprints = result_fingerprints(f"prompt for {task_id}", output)
    return TaskResult(
        task_id=task_id,
        path=f"{task_id}.py",
        ok=True,
        wall_clock_seconds=0.1,
        tool_calls=0,
        found_bug_ids=found_bug_ids,
        expected_bug_count=2,
        found_bug_count=len(found_bug_ids),
        success=len(found_bug_ids) == 2,
        output_quality=5 if len(found_bug_ids) == 2 else 3,
        output_text=output,
        **fingerprints,
    )


def test_output_equivalence_is_task_level_not_byte_level() -> None:
    reference = [
        make_result("task-1", ["bug-a", "bug-b"], "Bug A and Bug B."),
        make_result("task-2", ["bug-c"], "Bug C."),
    ]
    candidate = [
        make_result("task-1", ["bug-b", "bug-a"], "Found both issues with different wording."),
        make_result("task-2", ["bug-c"], "Bug C."),
    ]

    audit = output_equivalence_audit("isolated", reference, "shared", candidate)

    assert audit["task_equivalent"] is True
    assert audit["byte_identical"] is False
    assert audit["finding_set_match_rate"] == 1.0
    assert audit["exact_output_match_rate"] == 0.5
    assert audit["prompt_hash_match_rate"] == 1.0


def test_output_equivalence_detects_changed_findings() -> None:
    reference = [make_result("task-1", ["bug-a", "bug-b"], "Bug A and Bug B.")]
    candidate = [make_result("task-1", ["bug-a"], "Bug A only.")]

    audit = output_equivalence_audit("isolated", reference, "shared", candidate)

    assert audit["task_equivalent"] is False
    assert audit["finding_set_match_rate"] == 0.0
    assert audit["rows"][0]["reference_found_bug_ids"] == ["bug-a", "bug-b"]
    assert audit["rows"][0]["candidate_found_bug_ids"] == ["bug-a"]
