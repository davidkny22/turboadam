"""Smoke test: verify Triton kernel compilation works on this system."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def test_triton_add():
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    assert torch.allclose(out, x + y, atol=1e-5)
    print("Triton kernel compiled and ran successfully")


if __name__ == "__main__":
    test_triton_add()
