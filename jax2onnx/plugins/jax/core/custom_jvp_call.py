# Compatibility changes for JAX 0.6.0 and Flax 0.10.7; see NOTICE.
# file: jax2onnx/plugins/jax/core/custom_jvp_call.py
from typing import TYPE_CHECKING
import jax
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


# ---------- tiny custom-JVP function used by the testcase ----------
@jax.custom_jvp
def square(x):
    return x * x


@square.defjvp
def square_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return square(x), 2 * x * t


# -------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive="custom_jvp_call",
    jax_doc="Generic passthrough for custom JVP calls",
    onnx=[{"component": "CustomJvp", "doc": ""}],
    context="primitives.core",
    component="custom_jvp_generic",
    since="v0.7.1",
    testcases=[
        {
            "testcase": "custom_jvp_square",
            "callable": lambda x: square(x),
            "input_shapes": [(3,)],
        },
    ],
)
class GenericCustomJvpPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(*xs, **params):
        return params["out_abstract_vals"]

    def to_onnx(self, s, node_inputs, node_outputs, params):
        s._inline_jaxpr(params["call_jaxpr"], node_inputs, node_outputs)
