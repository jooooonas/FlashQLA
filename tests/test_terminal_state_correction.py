"""Hopper partition-state propagation against an independent FP64 recurrence."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]


def reference(buffer, matrix, mask, boundaries, initial, reverse, v_first):
    result = torch.empty_like(buffer, dtype=torch.float64)
    for sequence, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (torch.zeros_like(result[0]) if initial is None
                 else initial[sequence].double())
        for index in (reversed(range(start, end)) if reverse else range(start, end)):
            result[index] = state
            transform = matrix[index].double()
            if reverse:
                transform = transform.transpose(-1, -2)
            propagated = (state @ transform.transpose(-1, -2) if v_first
                          else transform @ state)
            state = buffer[index].double() + torch.where(
                mask[index, :, None, None], propagated, 0,
            )
    return result


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("v_first", [False, True])
@pytest.mark.parametrize("initial_state", [False, True])
@pytest.mark.parametrize("mixed_mask", [False, True])
def test_partition_recurrence(reverse, v_first, initial_state, mixed_mask):
    from flash_qla.ops.gated_delta_rule.chunk.hopper.cp_fwd import (
        correct_initial_states,
        correct_terminal_states,
    )

    torch.manual_seed(136)
    # Production uses FP32 backward buffers and BF16 forward buffers. Unequal
    # sequence lengths include the zero-iteration (single partition) case.
    dtype = torch.float32 if reverse else torch.bfloat16
    buffer = torch.randn(9, 3, 128, 128, device="cuda").to(dtype)
    matrix = (torch.randn_like(buffer.float()) * 0.01
              + 0.5 * torch.eye(128, device="cuda")).to(dtype)
    mask = (torch.rand(9, 3, device="cuda") > 0.5 if mixed_mask
            else torch.ones(9, 3, device="cuda", dtype=torch.bool))
    initial = (torch.randn(3, 3, 128, 128, device="cuda")
               if initial_state else None)
    boundaries = [0, 1, 4, 9]
    mapping = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    expected = reference(buffer, matrix, mask, boundaries, initial, reverse, v_first)
    function = correct_terminal_states if reverse else correct_initial_states
    result = function(initial, buffer, matrix, mask, mapping, state_v_first=v_first)
    tolerance = 5e-4 if reverse else 5e-3
    relative_error = (result.double() - expected).norm() / expected.norm()
    assert relative_error < tolerance, relative_error.item()
    for _ in range(5):
        repeated = function(initial, buffer, matrix, mask, mapping, state_v_first=v_first)
        assert torch.equal(result, repeated), "partition correction is nondeterministic"


def test_half_identity_backward_recurrence():
    from flash_qla.ops.gated_delta_rule.chunk.hopper.cp_fwd import correct_terminal_states

    torch.manual_seed(39)
    buffer = torch.randn(4, 16, 128, 128, device="cuda")
    matrix = (0.5 * torch.eye(128, device="cuda")).expand_as(buffer).contiguous()
    mask = torch.ones(4, 16, device="cuda", dtype=torch.bool)
    mapping = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    expected = torch.zeros_like(buffer, dtype=torch.float64)
    for index in (2, 1, 0):
        expected[index] = buffer[index + 1].double() + 0.5 * expected[index + 1]
    first = None
    for _ in range(20):
        result = correct_terminal_states(None, buffer, matrix, mask, mapping)
        torch.testing.assert_close(result.double(), expected, rtol=1e-6, atol=1e-6)
        if first is None:
            first = result
        else:
            assert torch.equal(result, first)
