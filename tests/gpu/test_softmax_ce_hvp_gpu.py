"""Run on NVIDIA Linux with this repository's patched Triton installed."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for HVP kernel tests", allow_module_level=True)

from ptx_gym import list_kernels, resolve_kernel
from ptx_gym.evaluation import OutputVerifier, Payload
from ptx_gym.helpers.kernels import extract_specification_from_operator
from ptx_gym.helpers.triton import extract_ptx
from ptx_gym.kernels.common.softmax_ce_hvp import SOFTMAX_CE_HVP_KERNELS


@pytest.mark.parametrize("kernel_cls", SOFTMAX_CE_HVP_KERNELS)
@pytest.mark.parametrize(
    "classes", [1, 2, 31, 32, 33, 255, 256, 257, 1000, 1024, 1025, 4096, 8192]
)
def test_triton_matches_torch_and_preserves_inputs(kernel_cls, classes):
    torch.manual_seed(42)
    op = kernel_cls(batch_size=3, num_classes=classes, autotune=False)
    inputs = op.get_random_input()
    copies = [x.clone() for x in inputs]
    actual, _ = op.forward_triton(inputs)
    expected = op.forward_torch(inputs)
    assert actual.dtype == torch.float32
    assert actual.shape == (3, classes)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for value, original in zip(inputs, copies):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("kernel_cls", SOFTMAX_CE_HVP_KERNELS)
def test_registry_metadata_and_same_operator_ptx_round_trip(kernel_cls):
    assert kernel_cls.__name__ in list_kernels()
    assert resolve_kernel(kernel_cls.__name__) is kernel_cls
    op = kernel_cls(autotune=False)
    assert (op.batch_size, op.num_classes) == (4096, 1000)
    spec = extract_specification_from_operator(op)
    assert spec.constexpr_values == {"BLOCK_SIZE": 1024}
    inputs = op.get_random_input()
    expected, compiled = op.forward_triton(inputs)
    ptx = extract_ptx(compiled)
    assert ptx and ".entry" in ptx
    payload = Payload(ptx=ptx, threads_x=op.num_warps * 32)
    op.set_ptx(payload.to_launch_dict())
    actual, _ = op.forward_triton(inputs, ptx=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kernel_cls", SOFTMAX_CE_HVP_KERNELS)
def test_zero_candidate_is_rejected_by_gym_verifier(kernel_cls, monkeypatch):
    op = kernel_cls(batch_size=4, num_classes=33, autotune=False)
    forward = op.forward_triton
    # Install a nonempty marker so OutputVerifier exercises the PTX branch.
    op.ptx = "test candidate"

    def zero_candidate(inputs, ptx=False):
        if ptx:
            return torch.zeros_like(inputs[0], dtype=torch.float32), None
        return forward(inputs)

    monkeypatch.setattr(op, "forward_triton", zero_candidate)
    verifier = OutputVerifier(
        iterations=1, rtol=op.verification_tolerance, atol=op.verification_tolerance
    )
    assert not verifier.verify(op)


@pytest.mark.parametrize("kernel_cls", SOFTMAX_CE_HVP_KERNELS)
def test_special_values_and_masks(kernel_cls):
    op = kernel_cls(batch_size=2, num_classes=33, autotune=False)
    logits = torch.full((2, 33), -10000.0, device="cuda", dtype=op.vector_dtype)
    logits[0, 0] = 10000.0
    first = logits.float().softmax(-1) if op.cached_probabilities else logits
    for vector in (
        torch.zeros_like(logits),
        torch.ones_like(logits),
        torch.randn_like(logits),
    ):
        actual, _ = op.forward_triton((first, vector))
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual, op.forward_torch((first, vector)), rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"num_classes": -1},
        {"num_classes": 8193},
    ],
)
def test_invalid_task_shape(kwargs):
    with pytest.raises(ValueError):
        SOFTMAX_CE_HVP_KERNELS[0](autotune=False, **kwargs)
