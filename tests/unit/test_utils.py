from batch_agent.compiler import TaskCompiler
from batch_agent.spec import BatchSpec
from batch_agent.utils import prefix_hash, strip_preamble_headers


def test_strip_preamble_header_hash_matches_prompt_without_header() -> None:
    prompt = (
        "x-anthropic-billing-header: cc_version=0.2.93; cch=abc123==\n"
        "You are a helpful assistant."
    )
    without_header = "You are a helpful assistant."

    assert strip_preamble_headers(prompt) == without_header
    assert prefix_hash(strip_preamble_headers(prompt)) == prefix_hash(without_header)


def test_strip_preamble_headers_removes_multiple_lines() -> None:
    prompt = (
        "x-anthropic-billing-header: session=one\n"
        "x-amz-bedrock-trace: session=two\n"
        "Keep this prompt."
    )

    assert strip_preamble_headers(prompt) == "Keep this prompt."


def test_strip_preamble_headers_no_preamble_unchanged() -> None:
    prompt = "System: keep every line.\nUser: do the work."

    assert strip_preamble_headers(prompt) == prompt


def test_batch_spec_strip_preamble_false_escape_hatch() -> None:
    spec = BatchSpec(
        task="x-anthropic-billing-header: session=abc\nDo {x}",
        inputs=[{"x": "work"}],
        strip_preamble=False,
    )
    plan = TaskCompiler().compile(spec)

    assert plan.shared.strip_preamble is False
    assert plan.shared.prefix.startswith("x-anthropic-billing-header")
