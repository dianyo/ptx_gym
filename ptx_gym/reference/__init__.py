"""Mathematical references that require Torch but do not import Triton."""

from .softmax_ce_hvp import softmax_ce_hvp, softmax_ce_hvp_from_probs

__all__ = ["softmax_ce_hvp", "softmax_ce_hvp_from_probs"]
