from __future__ import annotations

from pydantic import BaseModel

from batch_agent.repair import parse_and_validate_output


class Payload(BaseModel):
    name: str


def test_parse_and_validate_repairs_trailing_comma() -> None:
    output = parse_and_validate_output('Here is JSON: {"name": "Ada",}', Payload)

    assert output.name == "Ada"
