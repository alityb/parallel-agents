"""Tests for batch_agent.schema — tool schema generation from type annotations."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from batch_agent.schema import annotation_to_schema, build_tool_schemas
from batch_agent.tools import Tool, ToolDefinition


def test_primitives():
    assert annotation_to_schema(str) == {"type": "string"}
    assert annotation_to_schema(int) == {"type": "integer"}
    assert annotation_to_schema(float) == {"type": "number"}
    assert annotation_to_schema(bool) == {"type": "boolean"}


def test_optional():
    result = annotation_to_schema(Optional[str])
    assert result == {"anyOf": [{"type": "string"}, {"type": "null"}]}


def test_list_generic():
    result = annotation_to_schema(list[int])
    assert result == {"type": "array", "items": {"type": "integer"}}


def test_list_bare():
    result = annotation_to_schema(list)
    assert result == {"type": "array"}


def test_dict_generic():
    result = annotation_to_schema(dict[str, int])
    assert result == {"type": "object", "additionalProperties": {"type": "integer"}}


def test_pydantic_model():
    class Inner(BaseModel):
        x: int
        y: str

    result = annotation_to_schema(Inner)
    assert result["type"] == "object"
    assert "x" in result["properties"]
    assert "y" in result["properties"]


def test_build_tool_schemas_full():
    @Tool.define(name="test_schema_tool")
    async def my_tool(query: str, count: int = 5, tags: list[str] | None = None) -> str:
        """Search for something."""
        return ""

    schemas = build_tool_schemas({"test_schema_tool": Tool.registry["test_schema_tool"]})
    assert len(schemas) == 1
    s = schemas[0]
    assert s["name"] == "test_schema_tool"
    assert s["description"] == "Search for something."
    assert s["input_schema"]["required"] == ["query"]
    assert s["input_schema"]["properties"]["query"] == {"type": "string"}
    assert s["input_schema"]["properties"]["count"] == {"type": "integer", "description": "Default: 5"}
