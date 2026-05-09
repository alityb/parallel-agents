from __future__ import annotations

import json
import string
from typing import Any

from .spec import AgentJob, BatchSpec, ExecutionPlan, SharedContext
from .utils import extract_schema


SCHEMA_INSTRUCTION = (
    "Your final message must be a valid JSON object matching this schema. "
    "Do not add prose after the JSON."
)


class TaskCompiler:
    def compile(self, spec: BatchSpec) -> ExecutionPlan:
        fields = self._template_fields(spec.task)
        schema = self._schema_for(spec.output_schema)
        prefix = self._build_prefix(spec, fields, schema)
        jobs = [self._build_job(spec, index, dict(input_data), prefix) for index, input_data in enumerate(spec.inputs)]
        return ExecutionPlan(
            shared=SharedContext(
                prefix=prefix,
                schema=schema,
                hoisted_inputs=self._hoisted(spec, fields),
                strip_preamble=spec.strip_preamble,
            ),
            jobs=jobs,
            spec=spec,
        )

    def _build_job(self, spec: BatchSpec, index: int, input_data: dict[str, Any], prefix: str) -> AgentJob:
        try:
            prompt = spec.task.format(**input_data)
        except KeyError as exc:
            raise ValueError(f"input {index} is missing template variable {exc.args[0]!r}") from exc
        except Exception as exc:  # pragma: no cover - defensive formatting path
            raise ValueError(f"input {index} could not format task template: {exc}") from exc

        estimated = estimate_tokens(prefix) + estimate_tokens(prompt)
        oversized = estimated > spec.model_max_context - spec.min_response_tokens
        return AgentJob(
            job_id=f"job-{index}",
            index=index,
            input_data=input_data,
            prompt=prompt,
            estimated_prompt_tokens=estimated,
            oversized=oversized,
            max_turns=spec.max_turns,
        )

    def _build_prefix(self, spec: BatchSpec, fields: list[str], schema: dict[str, Any] | None) -> str:
        parts: list[str] = []
        if spec.system_prompt:
            parts.append(spec.system_prompt.strip())
        else:
            parts.append(self._text_before_first_field(spec.task).strip())

        hoisted = self._hoisted(spec, fields)
        if hoisted:
            rendered = json.dumps(hoisted, sort_keys=True, ensure_ascii=True)
            parts.append(f"Shared input values: {rendered}")

        if schema:
            parts.append(f"{SCHEMA_INSTRUCTION}\nSchema:\n{json.dumps(schema, sort_keys=True, ensure_ascii=True)}")

        return "\n\n".join(part for part in parts if part)

    def _hoisted(self, spec: BatchSpec, fields: list[str]) -> dict[str, Any]:
        if spec.no_hoist or not spec.inputs:
            return {}

        hoisted: dict[str, Any] = {}
        for field in fields:
            values = [item.get(field) for item in spec.inputs]
            if any(field not in item for item in spec.inputs):
                continue
            if all(value == values[0] for value in values):
                hoisted[field] = values[0]
        return hoisted

    def _schema_for(self, output_schema: Any | None) -> dict[str, Any] | None:
        return extract_schema(output_schema)

    def _template_fields(self, template: str) -> list[str]:
        fields: list[str] = []
        for _, field_name, _, _ in string.Formatter().parse(template):
            if field_name:
                fields.append(field_name.split(".", 1)[0].split("[", 1)[0])
        return fields

    def _text_before_first_field(self, template: str) -> str:
        parsed = string.Formatter().parse(template)
        text = []
        for literal_text, field_name, _, _ in parsed:
            text.append(literal_text)
            if field_name is not None:
                break
        return "".join(text)


def estimate_tokens(text: str) -> int:
    # Phase 0 deliberately avoids model-specific tokenizers; ~4 chars/token is good enough for routing flags.
    return max(1, (len(text) + 3) // 4) if text else 0
