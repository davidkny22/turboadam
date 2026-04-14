"""Tests for utils: block padding helpers and parameter routing.

Covers:
- pad_to_blocks: exact multiple, non-multiple, single element
- unpad_from_blocks: roundtrip correctness
- is_matrix_param: routing logic (ndim + numel threshold)
"""

import torch
import pytest
from turboadam.utils import (
    BLOCK_SIZE,
    MATRIX_NUMEL_THRESHOLD,
    pad_to_blocks,
    unpad_from_blocks,
    is_matrix_param,
)


# ---------------------------------------------------------------------------
# pad_to_blocks
# ---------------------------------------------------------------------------

class TestPadToBlocks:
    def test_exact_multiple_unchanged_values(self):
        """When length is already a multiple of block_size, values are preserved."""
        t = torch.arange(256, dtype=torch.float32)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        assert orig_len == 256
        assert padded.shape[0] == 256
        assert torch.equal(padded, t)

    def test_exact_multiple_no_extra_elements(self):
        """Exact multiple: padded length equals original length."""
        t = torch.ones(128, dtype=torch.float32)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        assert padded.shape[0] == orig_len == 128

    def test_non_multiple_padded_to_next_block(self):
        """Non-multiple length is padded up to the next multiple of block_size."""
        t = torch.ones(130, dtype=torch.float32)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        assert orig_len == 130
        assert padded.shape[0] == 256  # next multiple of 128

    def test_non_multiple_padding_is_zeros(self):
        """Padding elements must be zero."""
        t = torch.ones(130, dtype=torch.float32)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        assert torch.all(padded[130:] == 0.0)

    def test_non_multiple_original_values_preserved(self):
        """Original values must not be altered by padding."""
        t = torch.arange(200, dtype=torch.float32)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        assert torch.equal(padded[:200], t)

    def test_single_element(self):
        """A single-element tensor is padded to one full block."""
        t = torch.tensor([3.14], dtype=torch.float32)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        assert orig_len == 1
        assert padded.shape[0] == 128
        assert padded[0].item() == pytest.approx(3.14)
        assert torch.all(padded[1:] == 0.0)

    def test_custom_block_size(self):
        """Works with a block_size other than 128."""
        t = torch.ones(10, dtype=torch.float32)
        padded, orig_len = pad_to_blocks(t, block_size=16)
        assert orig_len == 10
        assert padded.shape[0] == 16

    def test_returns_tuple(self):
        """Return value must be a 2-tuple."""
        result = pad_to_blocks(torch.ones(5), block_size=128)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_padded_tensor_is_1d(self):
        """Output padded tensor must be 1-D."""
        t = torch.ones(50, dtype=torch.float32)
        padded, _ = pad_to_blocks(t, block_size=128)
        assert padded.ndim == 1


# ---------------------------------------------------------------------------
# unpad_from_blocks
# ---------------------------------------------------------------------------

class TestUnpadFromBlocks:
    def test_roundtrip_exact_multiple(self):
        """unpad(pad(t)) == t for exact-multiple length."""
        t = torch.randn(256)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        recovered = unpad_from_blocks(padded, orig_len)
        assert torch.equal(recovered, t)

    def test_roundtrip_non_multiple(self):
        """unpad(pad(t)) == t for non-multiple length."""
        t = torch.randn(130)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        recovered = unpad_from_blocks(padded, orig_len)
        assert torch.equal(recovered, t)

    def test_roundtrip_single_element(self):
        """unpad(pad(t)) == t for a single element."""
        t = torch.tensor([42.0])
        padded, orig_len = pad_to_blocks(t, block_size=128)
        recovered = unpad_from_blocks(padded, orig_len)
        assert torch.equal(recovered, t)

    def test_output_length_matches_original(self):
        """Output length equals original_length."""
        t = torch.ones(77)
        padded, orig_len = pad_to_blocks(t, block_size=128)
        recovered = unpad_from_blocks(padded, orig_len)
        assert recovered.shape[0] == 77


# ---------------------------------------------------------------------------
# is_matrix_param
# ---------------------------------------------------------------------------

class TestIsMatrixParam:
    def test_2d_large_is_matrix(self):
        """2-D tensor above threshold is a matrix param."""
        p = torch.empty(200, 200)  # 40_000 elements
        assert is_matrix_param(p) is True

    def test_2d_small_is_not_matrix(self):
        """2-D tensor at or below threshold is not a matrix param."""
        p = torch.empty(10, 10)  # 100 elements
        assert is_matrix_param(p) is False

    def test_1d_large_is_not_matrix(self):
        """1-D tensor, even if large, is not a matrix param."""
        p = torch.empty(20_000)
        assert is_matrix_param(p) is False

    def test_3d_large_is_matrix(self):
        """3-D (or higher) tensor above threshold qualifies."""
        p = torch.empty(10, 50, 30)  # 15_000 elements
        assert is_matrix_param(p) is True

    def test_2d_exactly_at_threshold_is_not_matrix(self):
        """Exactly MATRIX_NUMEL_THRESHOLD elements should NOT qualify (strictly greater)."""
        # 100 x 100 = 10_000 == MATRIX_NUMEL_THRESHOLD, should be False
        p = torch.empty(100, 100)
        assert is_matrix_param(p) is False

    def test_2d_one_above_threshold_is_matrix(self):
        """One element above threshold qualifies."""
        # Need numel > 10_000; 101 x 100 = 10_100
        p = torch.empty(101, 100)
        assert is_matrix_param(p) is True
