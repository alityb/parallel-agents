"""Tool schema generation — converts Python function signatures to JSON Schema (Anthropic format).

Handles: str, int, float, bool, list, list[T], dict, dict[str,T], Optional[T], 
Union types, Pydantic models, and unannotated params (default: string).
"""
from __future__ import annotations

import inspect
import typing
from typing import Any, get_args, get_origin

from .tools import ToolDefinition


def build_tool_schemas(tools: dict[str, ToolDefinition]) -> list[dict[str, Any]]:
    """Build Anthropic-format tool schemas from resolved tool definitions."""
    schemas: list[dict[str, Any]] = []
    for name, defn in tools.items():
        sig = inspect.signature(defn.func)
        # Use get_type_hints to resolve string annotations from `from __future__ import annotations`
        try:
            hints = typing.get_type_hints(defn.func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param_name in hints:
                prop = annotation_to_schema(hints[param_name])
            elif param.annotation is not inspect.Parameter.empty:
                prop = annotation_to_schema(param.annotation)
            else:
                prop = {"type": "string"}

            # If param has a default, it's optional
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                # Add default value description if it's not None
                if param.default is not None:
                    prop.setdefault("description", f"Default: {param.default!r}")

            properties[param_name] = prop

        schemas.append({
            "name": name,
            "description": (defn.func.__doc__ or f"Tool: {name}").strip(),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return schemas


def annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to JSON Schema."""
    # Handle None / NoneType
    if annotation is type(None):
        return {"type": "null"}

    # Primitives
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    # Get the origin type for generics
    origin = get_origin(annotation)
    args = get_args(annotation)

    # typing.Optional[X] is Union[X, None]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] → schema of X, but mark as nullable
            inner = annotation_to_schema(non_none[0])
            # JSON Schema nullable pattern
            return {"anyOf": [inner, {"type": "null"}]}
        else:
            # Union[A, B, ...] → anyOf
            return {"anyOf": [annotation_to_schema(a) for a in args]}

    # list / List[T]
    if origin is list or annotation is list:
        if args:
            return {"type": "array", "items": annotation_to_schema(args[0])}
        return {"type": "array"}

    # dict / Dict[str, T]
    if origin is dict or annotation is dict:
        if args and len(args) == 2:
            return {"type": "object", "additionalProperties": annotation_to_schema(args[1])}
        return {"type": "object"}

    # tuple
    if origin is tuple:
        if args:
            return {"type": "array", "items": [annotation_to_schema(a) for a in args], "minItems": len(args), "maxItems": len(args)}
        return {"type": "array"}

    # Pydantic models
    if hasattr(annotation, "model_json_schema"):
        return annotation.model_json_schema()
    if hasattr(annotation, "schema"):
        return annotation.schema()

    # Fallback: string
    return {"type": "string"}
