"""Tests for pre-execution validation that a node is actually executable.

validate_prompt rejects a node whose declared entry point does not resolve to a
real method (a V1 FUNCTION typo, or a V3 node missing its execute override) before
any node runs, attributing the error to the offending node.
"""
import asyncio

import nodes
from comfy_api.latest import io
from execution import node_not_executable_reason, validate_prompt


class _GoodV1Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Test"

    def run(self):
        return (None,)


class _TypoV1Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"  # method below is misspelled
    OUTPUT_NODE = True
    CATEGORY = "Test"

    def invvert(self):
        return (None,)


class _InstanceMethodV1Node:
    """Defines its FUNCTION method on the instance, as the runtime resolves it."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Test"

    def __init__(self):
        self.run = lambda: (None,)


def _v3_schema(node_id):
    return io.Schema(
        node_id=node_id,
        display_name=node_id,
        category="Test",
        inputs=[],
        outputs=[io.Image.Output()],
        is_output_node=True,
    )


class _GoodV3Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _v3_schema("GoodV3Node")

    @classmethod
    def execute(cls):
        return io.NodeOutput(None)


class _TypoV3Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _v3_schema("TypoV3Node")

    @classmethod
    def exicute(cls):  # typo: should be "execute"
        return io.NodeOutput(None)


def _register(class_type, class_def):
    nodes.NODE_CLASS_MAPPINGS[class_type] = class_def


def _validate(class_type):
    prompt = {"1": {"class_type": class_type, "inputs": {}}}
    return asyncio.run(validate_prompt("pid", prompt, None))


def test_good_node_passes():
    _register("GoodV1Node", _GoodV1Node)
    assert node_not_executable_reason(_GoodV1Node, "GoodV1Node") is None
    valid, _, _, _ = _validate("GoodV1Node")
    assert valid is True


def test_typo_node_rejected_with_node_error():
    _register("TypoV1Node", _TypoV1Node)
    valid, error, _, node_errors = _validate("TypoV1Node")
    assert valid is False
    assert error["type"] == "invalid_node_definition"
    assert node_errors["1"]["class_type"] == "TypoV1Node"
    assert node_errors["1"]["errors"][0]["type"] == "invalid_node_definition"
    assert "invert" in node_errors["1"]["errors"][0]["details"]


def test_instance_defined_method_not_false_positived():
    """A node whose method is defined in __init__ runs fine and must not be blocked."""
    _register("InstanceMethodV1Node", _InstanceMethodV1Node)
    assert node_not_executable_reason(_InstanceMethodV1Node, "InstanceMethodV1Node") is None
    valid, _, _, _ = _validate("InstanceMethodV1Node")
    assert valid is True


def test_good_v3_node_passes():
    _register("GoodV3Node", _GoodV3Node)
    assert node_not_executable_reason(_GoodV3Node, "GoodV3Node") is None
    valid, _, _, _ = _validate("GoodV3Node")
    assert valid is True


def test_typo_v3_node_rejected_with_node_error():
    _register("TypoV3Node", _TypoV3Node)
    valid, error, _, node_errors = _validate("TypoV3Node")
    assert valid is False
    assert error["type"] == "invalid_node_definition"
    assert node_errors["1"]["errors"][0]["type"] == "invalid_node_definition"
