# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""unfillable_arguments: the argument shapes the Gemini converter loses."""

from airc_core.toolschema import unfillable_arguments
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class _File(BaseModel):
    name: str
    content: str


class _Meta(BaseModel):
    title: str
    tags: dict[str, str]


def _tool(schema: type[BaseModel]) -> StructuredTool:
    async def run(**kwargs):
        return ""

    return StructuredTool.from_function(
        coroutine=run, name="t", description="t", args_schema=schema
    )


def test_a_dict_typed_argument_is_reported():
    class Args(BaseModel):
        files: dict[str, str]
        flags: list[str] = []

    assert unfillable_arguments(_tool(Args)) == ["files"]


def test_a_dict_nested_in_an_object_or_a_list_is_reported_with_its_path():
    class Args(BaseModel):
        meta: _Meta
        extra: list[dict[str, str]] = []

    assert unfillable_arguments(_tool(Args)) == ["meta.tags", "extra[]"]


def test_an_optional_dict_is_reported_once():
    class Args(BaseModel):
        env: dict[str, str] | None = None

    assert unfillable_arguments(_tool(Args)) == ["env"]


def test_declared_fields_and_lists_of_objects_pass():
    class Args(BaseModel):
        script: str
        files: list[_File] = []
        revisions: list[str] = []
        owner: _File | None = None

    assert unfillable_arguments(_tool(Args)) == []


def test_a_tool_without_arguments_passes():
    class Args(BaseModel):
        pass

    assert unfillable_arguments(_tool(Args)) == []
