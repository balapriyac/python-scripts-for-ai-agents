#!/usr/bin/env python3
"""
schema_from_function.py

Generate LLM tool schemas from real Python functions, so you never type
JSON schemas by hand again.

For every public function in a module, this script reads:

    - the function name            -> tool name
    - the docstring summary        -> tool description
    - "Args:" lines in the docstring -> parameter descriptions
    - type hints                   -> JSON Schema types
    - default values               -> "required" vs optional

Supported hints: str, int, float, bool, list[T], tuple, set, dict[str, T],
Optional[T], T | None, Union, Literal[...], Enum classes, datetime/date,
Path, and Annotated[T, "description"].

Docstring styles: Google ("Args:") and Sphinx (":param name: ...").

The output is a neutral "manifest" (see manifest_exporter.py to turn it into
OpenAI / Anthropic / MCP formats).

IMPORTANT: this script IMPORTS your module, which runs its top-level code.
Only point it at code you trust. To scan files without running them, use
ast_tool_scanner.py instead.

Usage:
    python schema_from_function.py my_tools.py
    python schema_from_function.py my_tools.py --output manifest.json
    python schema_from_function.py --demo
"""

import argparse
import datetime
import enum
import importlib.util
import inspect
import json
import pathlib
import re
import sys
import tempfile
import types
import typing

NONE_TYPE = type(None)
SECTION_ARGS = {"args", "arguments", "parameters", "params"}
SECTION_OTHER = {"returns", "return", "raises", "yields", "example", "examples", "note", "notes"}


# --------------------------------------------------------------------------
# Docstrings
# --------------------------------------------------------------------------

def parse_docstring(doc):
    """Return (description, {param_name: description}) from a Google or Sphinx docstring."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary, params = [], {}
    section, current, param_indent = None, None, None

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        header = stripped[:-1].lower() if stripped.endswith(":") else ""

        if indent == 0 and header in SECTION_ARGS:
            section, current, param_indent = "args", None, None
            continue
        if indent == 0 and header in SECTION_OTHER:
            section, current = "other", None
            continue

        sphinx = re.match(r":param\s+(?:\S+\s+)?(\w+):\s*(.*)", stripped)
        if sphinx:
            section, current = "sphinx", sphinx.group(1)
            params[current] = sphinx.group(2)
            continue
        if re.match(r":(returns?|raises?|rtype|type)\b", stripped):
            section, current = "other", None
            continue

        if section is None:
            summary.append(stripped)
        elif section == "args":
            if not stripped:
                continue
            if param_indent is None:
                param_indent = indent
            match = re.match(r"\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if match and indent <= param_indent:
                current = match.group(1)
                params[current] = match.group(2)
            elif current:
                params[current] += " " + stripped
        elif section == "sphinx" and current and stripped:
            params[current] += " " + stripped

    description = " ".join(" ".join(summary).split())
    return description, {k: " ".join(v.split()) for k, v in params.items()}


# --------------------------------------------------------------------------
# Type hints -> JSON Schema
# --------------------------------------------------------------------------

def jsonable(value):
    """True if the value can be written as strict JSON (rules out bytes, sets, NaN and so on)."""
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def json_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "null"


def make_nullable(schema):
    schema = dict(schema)
    if isinstance(schema.get("type"), str):
        schema["type"] = [schema["type"], "null"]
        if "enum" in schema and None not in schema["enum"]:
            schema["enum"] = schema["enum"] + [None]
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def annotation_to_schema(tp, warnings, where):
    origin, args = typing.get_origin(tp), typing.get_args(tp)

    if tp is inspect.Parameter.empty or tp is typing.Any:
        return {}
    if tp is None or tp is NONE_TYPE:
        return {"type": "null"}

    if origin is typing.Annotated:
        schema = annotation_to_schema(args[0], warnings, where)
        for meta in args[1:]:
            if isinstance(meta, str):
                return {**schema, "description": meta}
        return schema

    if origin is typing.Literal:
        if not jsonable(list(args)):
            warnings.append(f"{where}: Literal values can't be written as JSON, left unconstrained")
            return {}
        types_found = {json_type(v) for v in args}
        schema = {"enum": list(args)}
        if len(types_found) == 1:
            schema["type"] = types_found.pop()
        return schema

    if origin in (typing.Union, types.UnionType):
        members = [a for a in args if a is not NONE_TYPE]
        schemas = [annotation_to_schema(a, warnings, where) for a in members]
        has_null = len(members) < len(args)
        if len(schemas) == 1:
            return make_nullable(schemas[0]) if has_null else schemas[0]
        anyof = schemas + ([{"type": "null"}] if has_null else [])
        return {"anyOf": anyof}

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        values = [member.value for member in tp]
        if not jsonable(values):
            warnings.append(f"{where}: enum values can't be written as JSON, left unconstrained")
            return {}
        schema = {"enum": values}
        kinds = {json_type(v) for v in values}
        if len(kinds) == 1:
            schema["type"] = kinds.pop()
        return schema

    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}

    base = origin or tp
    if base in (list, set, frozenset, collections_abc("Sequence"), collections_abc("Iterable")):
        schema = {"type": "array"}
        if args:
            schema["items"] = annotation_to_schema(args[0], warnings, where)
        if base in (set, frozenset):
            schema["uniqueItems"] = True
        return schema
    if base is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": annotation_to_schema(args[0], warnings, where)}
        return {"type": "array",
                "prefixItems": [annotation_to_schema(a, warnings, where) for a in args],
                "minItems": len(args), "maxItems": len(args)}
    if base in (dict, collections_abc("Mapping")):
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = annotation_to_schema(args[1], warnings, where)
        return schema

    if tp is datetime.datetime:
        return {"type": "string", "format": "date-time"}
    if tp is datetime.date:
        return {"type": "string", "format": "date"}
    if isinstance(tp, type) and issubclass(tp, pathlib.PurePath):
        return {"type": "string"}

    warnings.append(f"{where}: unsupported type {tp!r}, left unconstrained")
    return {}


def collections_abc(name):
    import collections.abc
    return getattr(collections.abc, name)


# --------------------------------------------------------------------------
# Function -> tool
# --------------------------------------------------------------------------

def function_to_tool(func, warnings):
    name = func.__name__
    description, param_docs = parse_docstring(inspect.getdoc(func))
    if not description:
        warnings.append(f"{name}: no docstring, so the tool has no description")

    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception as exc:
        # One unresolvable hint shouldn't cost us all the others: resolve parameter by parameter.
        warnings.append(f"{name}: could not resolve all type hints ({exc}); resolving them one at a time")
        hints = {}
        for pname, raw in getattr(func, "__annotations__", {}).items():
            try:
                hints[pname] = eval(raw, func.__globals__) if isinstance(raw, str) else raw  # noqa: S307
            except Exception:
                pass  # left unconstrained; the warning above explains why

    properties, required = {}, []
    for pname, param in inspect.signature(func).parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            warnings.append(f"{name}: skipped *{pname}; tools need named parameters")
            continue

        annotation = hints.get(pname, param.annotation)
        schema = annotation_to_schema(annotation, warnings, f"{name}.{pname}")
        if "description" not in schema and pname in param_docs:
            schema["description"] = param_docs[pname]

        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            default = param.default.value if isinstance(param.default, enum.Enum) else param.default
            if default is not None:
                if jsonable(default):
                    schema["default"] = default
                else:
                    warnings.append(f"{name}.{pname}: default {default!r} can't be written as JSON, not recorded")
        properties[pname] = schema

    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "source": f"{pathlib.Path(inspect.getsourcefile(func) or '?').name}:{func.__code__.co_firstlineno}",
    }


def load_module(path):
    path = pathlib.Path(path).resolve()
    module_name = f"_tools_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # needed so type hints and dataclasses resolve
    spec.loader.exec_module(module)
    return module


def build_manifest(path):
    module = load_module(path)
    if hasattr(module, "__all__"):
        names = list(module.__all__)
    else:
        names = [n for n, obj in vars(module).items()
                 if inspect.isfunction(obj) and obj.__module__ == module.__name__ and not n.startswith("_")]
    warnings, tools = [], []
    for name in names:
        obj = getattr(module, name, None)
        if inspect.isfunction(obj):
            tools.append(function_to_tool(obj, warnings))
    return {"tools": tools}, warnings


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

SAMPLE_MODULE = '''
import datetime
import enum
from typing import Annotated, Literal, Optional


class Priority(enum.Enum):
    LOW = "low"
    HIGH = "high"


def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, for example "Paris".
        unit: Temperature unit to report in.
    """
    return f"21 degrees {unit} in {city}"


def search_orders(
    customer_id: int,
    status: Optional[str] = None,
    tags: list[str] | None = None,
    limit: Annotated[int, "Maximum number of orders to return"] = 10,
) -> list:
    """Find orders for a customer.

    Args:
        customer_id: Numeric customer ID.
        status: Only return orders with this status.
        tags: Only return orders that carry all of these tags.
    """
    return []


def create_ticket(title: str, priority: Priority = Priority.LOW, due: datetime.date | None = None) -> dict:
    """Open a support ticket.

    :param title: Short title for the ticket.
    :param priority: How urgent the ticket is.
    :param due: Optional due date.
    """
    return {}


def _internal_helper(x):
    return x
'''


def main():
    parser = argparse.ArgumentParser(description="Generate LLM tool schemas from Python functions.")
    parser.add_argument("module", nargs="?", help="Path to a .py file containing your tool functions")
    parser.add_argument("--output", default=None, help="Write the manifest to this file (default: print)")
    parser.add_argument("--demo", action="store_true", help="Run on a built-in sample module")
    args = parser.parse_args()

    if args.demo:
        folder = pathlib.Path(tempfile.mkdtemp())
        target = folder / "sample_tools.py"
        target.write_text(SAMPLE_MODULE, encoding="utf-8")
        print(f"DEMO: generated from a sample module ({target})", file=sys.stderr)
    elif args.module:
        target = args.module
    else:
        parser.error("give a module path, or use --demo")

    manifest, warnings = build_manifest(target)
    text = json.dumps(manifest, indent=2)

    if args.output:
        pathlib.Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {len(manifest['tools'])} tool(s) to {args.output}", file=sys.stderr)
    else:
        print(text)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
