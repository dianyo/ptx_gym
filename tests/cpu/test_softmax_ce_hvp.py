"""Independent mathematical checks; this module must not require Triton."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ptx_gym.reference import softmax_ce_hvp, softmax_ce_hvp_from_probs

DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def autograd_hvp(logits, vector, labels=None, reduction="sum"):
    z = logits.double().detach().requires_grad_()
    if labels is None:
        labels = torch.zeros(z.shape[0], dtype=torch.long)
    loss = F.cross_entropy(z, labels, reduction=reduction)
    gradient = torch.autograd.grad(loss, z, create_graph=True)[0]
    return torch.autograd.grad((gradient * vector.double()).sum(), z)[0]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("classes", [1, 2, 31, 33, 255, 1000, 8192])
def test_matches_float64_autograd(dtype, classes):
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(3, classes, generator=generator).to(dtype)
    vector = torch.randn(3, classes, generator=generator).to(dtype)
    expected = autograd_hvp(logits, vector)
    fused = softmax_ce_hvp(logits, vector)
    cached = softmax_ce_hvp_from_probs(logits.float().softmax(-1), vector)
    assert fused.dtype == cached.dtype == torch.float32
    assert fused.shape == logits.shape
    torch.testing.assert_close(fused.double(), expected, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(cached, fused, rtol=0, atol=0)


def test_small_explicit_hessian_and_loss_conventions():
    z = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -2.0]])
    v = torch.tensor([[2.0, -1.0, 0.5], [0.3, 0.7, -0.1]])
    p = z.double().softmax(-1)
    dense = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    actual = softmax_ce_hvp(z, v).double()
    torch.testing.assert_close(actual, (dense @ v.double().unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(actual, autograd_hvp(z, v, torch.tensor([2, 1])))
    torch.testing.assert_close(actual / len(z), autograd_hvp(z, v, reduction="mean"))


def test_curvature_invariants_and_inputs_unchanged():
    generator = torch.Generator().manual_seed(7)
    z, u, v = [torch.randn(4, 33, generator=generator) for _ in range(3)]
    copies = [x.clone() for x in (z, u, v)]
    hv = softmax_ce_hvp(z, v)
    hu = softmax_ce_hvp(z, u)
    torch.testing.assert_close(
        softmax_ce_hvp(z, torch.zeros_like(v)), torch.zeros_like(v)
    )
    torch.testing.assert_close(
        softmax_ce_hvp(z, torch.ones_like(v)), torch.zeros_like(v), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(softmax_ce_hvp(z + 20.0, v), hv, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(hv.sum(-1), torch.zeros(len(z)), atol=1e-6, rtol=0)
    torch.testing.assert_close((u * hv).sum(), (v * hu).sum(), atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(softmax_ce_hvp(z, u + v), hu + hv, atol=1e-6, rtol=2e-5)
    assert torch.all((v * hv).sum(-1) >= -1e-6)
    for original, copy in zip((z, u, v), copies):
        torch.testing.assert_close(original, copy, rtol=0, atol=0)
    # The Gym tolerance must not accept a kernel that simply writes zeros.
    assert not torch.allclose(hv, torch.zeros_like(hv), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extreme_finite_logits(dtype):
    z = torch.tensor(
        [[10000.0, 9999.0, -10000.0], [-10000.0, -10000.0, -10000.0]], dtype=dtype
    )
    v = torch.tensor([[2.0, -1.0, 0.5], [0.3, 0.7, -0.1]], dtype=dtype)
    actual = softmax_ce_hvp(z, v)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.double(), autograd_hvp(z, v), rtol=2e-5, atol=1e-6
    )


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3,), (1, 2, 3)])
def test_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        softmax_ce_hvp(torch.empty(shape), torch.empty(shape))


def test_rejects_mismatch_layout_and_dtype():
    with pytest.raises(ValueError, match="matching"):
        softmax_ce_hvp(torch.ones(2, 3), torch.ones(2, 4))
    with pytest.raises(ValueError, match="contiguous"):
        softmax_ce_hvp(torch.ones(2, 3).T, torch.ones(2, 3).T)
    with pytest.raises(TypeError, match="matching dtypes"):
        softmax_ce_hvp(torch.ones(2, 3), torch.ones(2, 3).half())
    with pytest.raises(TypeError, match="vectors"):
        softmax_ce_hvp(torch.ones(2, 3).double(), torch.ones(2, 3).double())
    with pytest.raises(TypeError, match="probabilities"):
        softmax_ce_hvp_from_probs(torch.ones(2, 3).half(), torch.ones(2, 3).half())
    with pytest.raises(TypeError, match="tensors"):
        softmax_ce_hvp([[1.0]], torch.ones(1, 1))
    with pytest.raises(ValueError, match="same device"):
        softmax_ce_hvp(torch.ones(2, 3), torch.ones(2, 3, device="meta"))


def test_reference_import_does_not_import_triton():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ptx_gym.reference; assert 'triton' not in sys.modules",
        ],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    # Make it explicit that no Triton stub is part of this CPU test suite.
    if importlib.util.find_spec("triton") is None:
        assert "triton" not in sys.modules
