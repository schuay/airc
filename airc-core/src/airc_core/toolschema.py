# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Tool arguments a Gemini function declaration can carry.

langchain-google-genai does not pass a tool's JSON schema through. It rebuilds
the declaration from a fixed set of keys and drops `additionalProperties`, so a
dict-typed argument reaches the model as an object with no declared members,
or, nested inside another object, as a plain string. The model has nothing to
fill, sends an empty object or a string on every call, and the tool rejects
each one; no layer reports the mismatch. The OpenAI and Anthropic adapters send
the schema unchanged, so the same tool works there and the failure shows only
on the Gemini stack.

`unfillable_arguments` runs a tool through that converter and names every
object in the tool's own schema that the declaration no longer describes.
Each component's tests run it over the tools the component builds, so an
argument shape the converter cannot carry fails the suite rather than a
deployment.
"""

from __future__ import annotations

from google.genai import types
from langchain_google_genai._function_utils import (
    convert_to_genai_function_declarations,
)
from pydantic import BaseModel


def unfillable_arguments(tool) -> list[str]:
    """Paths of the arguments of `tool` whose schema describes an object with
    members but whose Gemini declaration does not. Nested fields are dotted,
    list items carry `[]`. Empty when the model can fill every argument."""
    source = tool.args_schema
    if isinstance(source, type) and issubclass(source, BaseModel):
        source = source.model_json_schema()
    if not isinstance(source, dict):
        return []
    (declaration,) = convert_to_genai_function_declarations([tool])[
        0
    ].function_declarations
    found: list[str] = []
    _walk(source, declaration.parameters, "", source.get("$defs") or {}, found)
    return found


def _walk(
    node: dict, sent: types.Schema | None, path: str, defs: dict, found: list[str]
) -> None:
    if "$ref" in node:
        node = defs.get(node["$ref"].rsplit("/", 1)[-1], {})
    if alternatives := node.get("anyOf"):
        for alternative in alternatives:
            if alternative.get("type") == "null":
                continue
            _walk(alternative, _matching(alternative, sent), path, defs, found)
        return
    members = node.get("properties") or {}
    extra = node.get("additionalProperties")
    if node.get("type") == "object" and (members or extra):
        described = (
            sent is not None
            and sent.type == types.Type.OBJECT
            and bool(sent.properties or sent.additional_properties)
        )
        if not described:
            # The root is the tool's own argument list; the converter always
            # sends it as an object, so a miss there is a missing argument.
            found.append(path or "*")
            return
    for name, child in members.items():
        child_sent = (sent.properties or {}).get(name) if sent else None
        _walk(child, child_sent, f"{path}.{name}" if path else name, defs, found)
    if isinstance(extra, dict):
        extra_sent = sent.additional_properties if sent else None
        if not isinstance(extra_sent, types.Schema):
            extra_sent = None
        _walk(extra, extra_sent, f"{path}.*" if path else "*", defs, found)
    if isinstance(node.get("items"), dict):
        _walk(node["items"], sent.items if sent else None, f"{path}[]", defs, found)


def _matching(alternative: dict, sent: types.Schema | None) -> types.Schema | None:
    """The declaration node for one anyOf alternative: the alternative of the
    same type when the converter kept the union, else the collapsed node."""
    if sent is None or not sent.any_of:
        return sent
    wanted = str(alternative.get("type", "")).upper()
    for candidate in sent.any_of:
        if candidate.type == wanted:
            return candidate
    return sent
