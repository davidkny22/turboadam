from __future__ import annotations

import pytest
import torch

from turboadam.utils import (
    ceil_div,
    finite_scalar,
    is_matrix_param,
    pad_to_blocks,
    state_tensor_bytes,
    unpad_from_blocks,
    validate_block_size,
)


@pytest.mark.parametrize("length", [1, 127, 128, 129, 256])
def test_block_padding_roundtrip(length: int) -> None:
    source = torch.arange(length, dtype=torch.float32)
    padded, original_length = pad_to_blocks(source, 128)
    assert padded.numel() % 128 == 0
    assert original_length == length
    assert torch.equal(unpad_from_blocks(padded, original_length), source)


def test_tensor_pad_value_avoids_scalar_extraction() -> None:
    source = torch.arange(7, dtype=torch.float32)
    padded, _ = pad_to_blocks(source, 8, pad_value=torch.tensor(3.5))
    assert padded[-1].item() == 3.5


def test_block_and_scalar_validation() -> None:
    assert ceil_div(129, 128) == 2
    assert finite_scalar(2, "value", non_negative=True) == 2.0
    with pytest.raises(TypeError):
        validate_block_size(True)
    with pytest.raises(ValueError):
        validate_block_size(0)
    with pytest.raises(ValueError):
        finite_scalar(float("nan"), "value")


def test_matrix_parameter_routing() -> None:
    assert is_matrix_param(torch.empty(101, 100))
    assert not is_matrix_param(torch.empty(100, 100))
    assert not is_matrix_param(torch.empty(20_000))


def test_state_byte_count_deduplicates_shared_storage() -> None:
    tensor = torch.zeros(17)
    assert state_tensor_bytes({"a": tensor, "b": [tensor]}) == 17 * 4
