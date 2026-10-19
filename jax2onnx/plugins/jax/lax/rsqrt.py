# Adapted from enpasos/jax2onnx 0.17.1 (Apache-2.0) for the 0.8.1 plugin API.
from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.rsqrt_p.name, component="rsqrt", context="primitives.lax",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rsqrt.html",
    onnx=[{"component": "Sqrt"}, {"component": "Div"}],
    testcases=[
        {"testcase": "rsqrt_scalar", "callable": lambda x: jax.lax.rsqrt(jax.numpy.abs(x) + 0.5), "input_shapes": [()]},
        {"testcase": "rsqrt_batch", "callable": lambda x: jax.lax.rsqrt(jax.numpy.abs(x) + 0.5), "input_shapes": [("B", 7)]},
    ],
)
class RsqrtPlugin(PrimitiveLeafPlugin):
    def to_onnx(self, converter: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params) -> None:
        source = converter.get_name(node_inputs[0])
        intermediate = converter.get_unique_name("rsqrt_sqrt")
        output = converter.get_var_name(node_outputs[0])
        aval = node_inputs[0].aval
        one = converter.get_constant_name(np.asarray(1, dtype=aval.dtype))
        converter.add_shape_info(intermediate, aval.shape, aval.dtype)
        converter.add_node(helper.make_node("Sqrt", [source], [intermediate]))
        converter.add_node(helper.make_node("Div", [one, intermediate], [output]))
        converter.add_shape_info(output, aval.shape, aval.dtype)
