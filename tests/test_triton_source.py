from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path


def test_triton_module_imports_under_frontend_stub() -> None:
    path = Path(__file__).parents[1] / "src" / "turboadam" / "triton_kernels.py"
    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_language.constexpr = object()
    fake_triton.language = fake_language
    fake_triton.jit = lambda function: function
    fake_triton.cdiv = lambda a, b: (a + b - 1) // b

    old_triton = sys.modules.get("triton")
    old_language = sys.modules.get("triton.language")
    sys.modules["triton"] = fake_triton
    sys.modules["triton.language"] = fake_language
    try:
        spec = importlib.util.spec_from_file_location("turboadam._triton_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        assert callable(module.triton_fused_ustate_adamw_step)
        assert module.triton_supports_block_size(128)
        assert not module.triton_supports_block_size(96)
    finally:
        if old_triton is None:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = old_triton
        if old_language is None:
            sys.modules.pop("triton.language", None)
        else:
            sys.modules["triton.language"] = old_language


def test_fused_wrapper_allocates_no_full_size_workspace() -> None:
    path = Path(__file__).parents[1] / "src" / "turboadam" / "triton_kernels.py"
    source = path.read_text()
    tree = ast.parse(source)
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "triton_fused_ustate_adamw_step"
    )
    text = ast.get_source_segment(source, wrapper)
    assert "torch.empty" not in text
    assert "torch.zeros" not in text
    assert "_finalize_ustate_scale_kernel" in text
    assert "rand_buf" not in source
