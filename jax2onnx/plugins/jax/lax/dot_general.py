# Compatibility changes for JAX 0.6.0 and Flax 0.10.7; see NOTICE.
# file: jax2onnx/plugins/jax/lax/dot_general.py

from typing import TYPE_CHECKING
from string import ascii_letters

import numpy as np
import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.dot_general_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="dot_general",
    testcases=[
        {
            "testcase": "dot_general",
            "callable": lambda x1, x2: jax.lax.dot_general(
                x1, x2, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
        },
        {
            "testcase": "dot_general_lhs1_rhs1",
            "callable": lambda x1, x2: jax.lax.dot_general(
                x1, x2, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
        },
    ],
)
class DotGeneralPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.dot_general to ONNX MatMul.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        lhs_var, rhs_var = node_inputs
        out_var = node_outputs[0]

        lhs_name = s.get_name(lhs_var)
        rhs_name = s.get_name(rhs_var)
        out_name = s.get_var_name(out_var)

        # ONNX contractions require one element type; JAX permits promoted operands.
        operand_names = [lhs_name, rhs_name]
        for index, var in enumerate(node_inputs):
            if var.aval.dtype != out_var.aval.dtype:
                name = s.get_unique_name('dot_operand')
                s.add_node(helper.make_node('Cast', [operand_names[index]], [name], to=s.builder._numpy_dtype_to_onnx(out_var.aval.dtype)))
                s.add_shape_info(name, var.aval.shape, out_var.aval.dtype)
                operand_names[index] = name
        lhs_name, rhs_name = operand_names

        out_shape = out_var.aval.shape

        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = params[
            "dimension_numbers"
        ]

        matrix_inputs = len(lhs_var.aval.shape) == len(rhs_var.aval.shape) == 2 and not lhs_batch

        # Standard matrix multiplication → use Gemm so ORT won't fuse into float-only FusedMatMul
        if matrix_inputs and lhs_contract == (1,) and rhs_contract == (0,):
            zero_const = s.get_constant_name(np.array(0, dtype=out_var.aval.dtype))
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[lhs_name, rhs_name, zero_const],
                outputs=[out_name],
                name=s.get_unique_name("dot_general_gemm"),
                alpha=1.0,
                beta=0.0,
            )
            s.add_node(gemm_node)
            s.add_shape_info(out_name, out_shape, out_var.aval.dtype)

        # Contraction of the last dimension of both inputs → transpose then Gemm
        elif matrix_inputs and lhs_contract == (1,) and rhs_contract == (1,):
            transposed_rhs_name = s.get_unique_name("transposed_rhs")
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[rhs_name],
                    outputs=[transposed_rhs_name],
                    perm=[1, 0],
                    name=s.get_unique_name("transpose_rhs"),
                )
            )
            transposed_shape = (rhs_var.aval.shape[1], rhs_var.aval.shape[0])
            s.add_shape_info(transposed_rhs_name, transposed_shape, out_var.aval.dtype)

            zero_const = s.get_constant_name(np.array(0, dtype=out_var.aval.dtype))
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[lhs_name, transposed_rhs_name, zero_const],
                outputs=[out_name],
                name=s.get_unique_name("dot_general_gemm"),
                alpha=1.0,
                beta=0.0,
            )
            s.add_node(gemm_node)
            s.add_shape_info(out_name, out_shape, out_var.aval.dtype)
        else:
            # JAX orders results as batch axes, free lhs axes, then free rhs axes.
            lhs_labels = list(ascii_letters[:len(lhs_var.aval.shape)])
            rhs_labels = list(ascii_letters[len(lhs_labels):len(lhs_labels) + len(rhs_var.aval.shape)])
            for left, right in zip(lhs_batch + lhs_contract, rhs_batch + rhs_contract):
                rhs_labels[right] = lhs_labels[left]
            output_labels = [lhs_labels[axis] for axis in lhs_batch]
            output_labels += [label for axis, label in enumerate(lhs_labels) if axis not in lhs_batch + lhs_contract]
            output_labels += [label for axis, label in enumerate(rhs_labels) if axis not in rhs_batch + rhs_contract]
            equation = ''.join(lhs_labels) + ',' + ''.join(rhs_labels) + '->' + ''.join(output_labels)
            s.add_node(helper.make_node('Einsum', [lhs_name, rhs_name], [out_name], equation=equation))
            s.add_shape_info(out_name, out_shape, out_var.aval.dtype)
