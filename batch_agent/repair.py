from __future__ import annotations

import json
import re
from typing import Any


class OutputValidationError(ValueError):
    pass


def parse_and_validate_output(content: str, schema_model: Any | None = None) -> Any:
    if schema_model is None:
        return content

    candidate = extract_json_object(content)
    data = loads_with_repair(candidate)

    if hasattr(schema_model, "model_validate"):
        return schema_model.model_validate(data)
    if hasattr(schema_model, "parse_obj"):
        return schema_model.parse_obj(data)
    return data


def extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise OutputValidationError("model response did not contain a JSON object")
    return content[start : end + 1]


def loads_with_repair(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        repaired = _simple_json_repair(content)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise OutputValidationError(f"model response was not valid JSON: {exc}") from exc


def _simple_json_repair(content: str) -> str:
    repaired = content.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired
