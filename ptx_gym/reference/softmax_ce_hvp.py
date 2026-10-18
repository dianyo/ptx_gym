"""FP32 reference actions of the unweighted cross-entropy logit Hessian.

Inputs are finite, contiguous, nonempty [batch, classes] tensors. Cached
probabilities must be normalized softmax probabilities in FP32. Value-domain
checks are left to callers to avoid synchronizing CUDA on every invocation.
"""

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
DEFAULT_BATCH_SIZE = 4096
DEFAULT_NUM_CLASSES = 1000
MAX_CLASSES = 8192


def _validate_inputs(
    first: torch.Tensor, vector: torch.Tensor, *, cached: bool
) -> None:
    if not isinstance(first, torch.Tensor) or not isinstance(vector, torch.Tensor):
        raise TypeError("HVP inputs must be Torch tensors.")
    if first.ndim != 2 or vector.ndim != 2:
        raise ValueError("HVP inputs must have rank 2: [batch, classes].")
    if first.shape != vector.shape or first.numel() == 0:
        raise ValueError("HVP inputs must have matching, nonempty shapes.")
    if first.device != vector.device:
        raise ValueError("HVP inputs must be on the same device.")
    if not first.is_contiguous() or not vector.is_contiguous():
        raise ValueError("HVP inputs must be contiguous.")
    if vector.dtype not in SUPPORTED_DTYPES:
        raise TypeError("HVP vectors must use float32, float16 or bfloat16.")
    if cached:
        if first.dtype != torch.float32:
            raise TypeError("Cached probabilities must use float32.")
    elif first.dtype != vector.dtype:
        raise TypeError("Fused HVP logits and vectors must have matching dtypes.")


def softmax_ce_hvp(logits: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Return H_z @ vector in FP32, without batch-mean scaling or labels."""
    _validate_inputs(logits, vector, cached=False)
    probability = torch.softmax(logits.float(), dim=-1)
    return softmax_ce_hvp_from_probs(probability, vector)


def softmax_ce_hvp_from_probs(
    probability: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Return p * (v - sum(p*v)) using supplied FP32 probabilities.

    This does not recompute or renormalize p. Keep p fixed across CG products
    at a fixed optimization point; refresh it when that point changes.
    """
    _validate_inputs(probability, vector, cached=True)
    vector_fp32 = vector.float()
    projection = torch.sum(probability * vector_fp32, dim=-1, keepdim=True)
    return probability * (vector_fp32 - projection)
