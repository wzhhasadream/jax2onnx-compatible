# Compatibility changes for JAX 0.6.0 and Flax 0.10.7; see NOTICE.
"""
Monkey Patching Utilities for JAX2ONNX

This module provides utilities for temporarily monkey patching JAX functions and
classes to enable capture and conversion of JAX operations to ONNX format.
These utilities are primarily used during the tracing phase of JAX to ONNX conversion.
"""

import contextlib
import inspect
from contextvars import ContextVar
from functools import wraps
from jax import lax
from jax._src.core import nonempty_axis_env, unsafe_am_i_under_a_vmap
from typing import Any, Callable, Generator, Mapping

from jax2onnx.plugin_system import (
    FunctionPlugin,
    ExamplePlugin,
    PrimitiveLeafPlugin,
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)

_native_batching = ContextVar('jax2onnx_native_batching', default=False)


@contextlib.contextmanager
def temporary_monkey_patches(
    allow_function_primitives: bool = False,
) -> Generator[None, None, None]:
    """
    Context manager that temporarily patches JAX functions and classes.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(_temporary_patch(lax, 'scan', lambda original: original))
        registries: list[
            Mapping[str, FunctionPlugin | ExamplePlugin | PrimitiveLeafPlugin]
        ] = [PLUGIN_REGISTRY]
        if allow_function_primitives:
            registries.append(ONNX_FUNCTION_PLUGIN_REGISTRY)

        for registry in registries:
            for plugin in registry.values():
                if not hasattr(plugin, "get_patch_params"):
                    continue
                try:
                    patch_params = plugin.get_patch_params()
                except Exception:
                    continue

                # Normalize to a list of (target, attr, patch_func) tuples
                if isinstance(patch_params, tuple):
                    patch_params = [patch_params]
                for target, attr, patch_func in patch_params:
                    stack.enter_context(_temporary_patch(target, attr, patch_func))

        yield


@contextlib.contextmanager
def _temporary_patch(
    target: Any, attr: str, patch_func: Callable
) -> Generator[None, None, None]:
    """
    Internal helper that temporarily patches a single attribute.

    This context manager saves the original attribute value, replaces it with
    the patched version, and ensures the original is restored when the context exits.

    Args:
        target: The object to patch (class, module, etc.)
        attr: The attribute name to patch
        patch_func: The function that produces the patch or is the patch itself

    Yields:
        None: A context where the patch is active
    """
    # Save the original attribute
    original = getattr(target, attr)

    # Apply the patch - either directly or by calling with the original
    # We determine which approach to use by checking if patch_func accepts parameters
    patched = (
        patch_func(original)
        if inspect.signature(patch_func).parameters
        else patch_func()
    )
    lowered_call = patched
    native_batching = not getattr(target, '__module__', '').startswith('equinox')

    @wraps(lowered_call)
    def patched(*args, **kwargs):
        # Preserve native batching rules, including scans nested inside a vmap.
        if not native_batching:
            return lowered_call(*args, **kwargs)
        if _native_batching.get():
            return original(*args, **kwargs)
        if nonempty_axis_env() or unsafe_am_i_under_a_vmap():
            token = _native_batching.set(True)
            try:
                return original(*args, **kwargs)
            finally:
                _native_batching.reset(token)
        return lowered_call(*args, **kwargs)

    setattr(target, attr, patched)

    try:
        yield
    finally:
        # Restore the original attribute when done
        setattr(target, attr, original)
