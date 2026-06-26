"""Regression tests for scheduler resilience to malformed nodes.

A node whose FUNCTION points at a method that does not exist (e.g. a typo in a
custom node) used to raise inside the scheduling heuristic, escaping the prompt
worker's error handling and silently killing the worker thread. Scheduling must
instead either proceed (so the error surfaces through normal execution) or report
the failure as an execution error.
"""
import asyncio

import nodes
from comfy_execution.graph import DynamicPrompt, ExecutionList


class _MalformedV1Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"  # the actual method below is misspelled
    OUTPUT_NODE = True
    CATEGORY = "Test"

    def invvert(self):
        return (None,)


class _RaisingDescriptor:
    def __get__(self, obj, owner):
        raise RuntimeError("schema error")


class _SchemaRaisesNode:
    """A node whose schema-derived attribute access raises, as a broken V3 node would."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    OUTPUT_NODE = _RaisingDescriptor()
    CATEGORY = "Test"

    def run(self):
        return (None,)


class _FakeOutputCache:
    def all_node_ids(self):
        return set()

    async def get(self, node_id):
        return None


def _make_execution_list(class_type, class_def):
    nodes.NODE_CLASS_MAPPINGS[class_type] = class_def
    prompt = {"1": {"class_type": class_type, "inputs": {}}}
    execution_list = ExecutionList(DynamicPrompt(prompt), _FakeOutputCache())
    execution_list.add_node("1")
    return execution_list


def test_malformed_function_does_not_crash_scheduler():
    """A FUNCTION-typo node schedules without raising; the error surfaces later."""
    execution_list = _make_execution_list("MalformedV1Node", _MalformedV1Node)
    node_id, error, ex = asyncio.run(execution_list.stage_node_execution())
    assert ex is None
    assert error is None
    assert node_id == "1"


def test_schema_attribute_error_does_not_crash_scheduler():
    """A node whose attribute access raises during heuristics still schedules."""
    execution_list = _make_execution_list("SchemaRaisesNode", _SchemaRaisesNode)
    node_id, error, ex = asyncio.run(execution_list.stage_node_execution())
    assert ex is None
    assert error is None
    assert node_id == "1"


def test_pick_node_failure_is_reported_not_raised():
    """An unexpected scheduling error is returned as an error, not raised."""
    execution_list = _make_execution_list("MalformedV1Node", _MalformedV1Node)

    def raise_on_pick(_available):
        raise RuntimeError("boom")

    execution_list.ux_friendly_pick_node = raise_on_pick
    node_id, error, ex = asyncio.run(execution_list.stage_node_execution())
    assert node_id is None
    assert isinstance(ex, RuntimeError)
    assert error["node_id"] == "1"
    assert error["exception_type"] == "RuntimeError"
    assert error["exception_message"] == "boom"
    assert error["traceback"]
