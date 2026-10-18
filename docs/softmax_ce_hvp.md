# Softmax cross-entropy Hessian-vector products

This contribution adds a logit-space curvature operation to PTX Gym. Start
with the [architecture walkthrough](architecture.md) for the harness, or use
the [runbook](run_softmax_ce_hvp.md) to run the code.

## 1. Derive the operation

For one example, let `z` be a length-`C` vector of logits and `y` a normalized
target distribution (a one-hot label is the usual case). Unweighted
cross-entropy is

\[
\ell(z,y)=-\sum_i y_i\log p_i
=\log\sum_j e^{z_j}-\sum_i y_i z_i,
\qquad p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

Its gradient is `p - y`. Differentiating a probability gives

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),
\qquad H_z=\operatorname{diag}(p)-pp^T.
\]

For a supplied direction `v`, multiply without materializing that `C×C`
matrix:

\[
(H_zv)_i=p_i v_i-p_i\sum_j p_jv_j
=p_i(v_i-s),\qquad s=\sum_j p_jv_j.
\]

The implementation is linear in `C` and needs only a row reduction and
elementwise operations after probabilities are known. It is more than a
Hadamard multiply: every output depends on the shared scalar `s`.

For a batch, the loss is `L = sum_b ell(z_b, y_b)`. Rows are independent
Hessian blocks. The output has shape `[B, C]` and is **not divided by B**.
For batch-mean cross-entropy, divide this output by `B` outside the kernel.
Labels are absent from the interface because the second derivative of
`-y^T z` is zero. Class weights, ignored samples, masking and other loss
conventions are not included in these tasks.

## 2. Two workloads, three input precisions

| Variant | First input | Direction input | Output |
| --- | --- | --- | --- |
| Fused Float32 | FP32 logits | FP32 | FP32 |
| Fused Float16 | FP16 logits | FP16 | FP32 |
| Fused BFloat16 | BF16 logits | BF16 | FP32 |
| Cached Float32 | FP32 probabilities | FP32 | FP32 |
| Cached Float16 | FP32 probabilities | FP16 | FP32 |
| Cached BFloat16 | FP32 probabilities | BF16 | FP32 |

Registered identifiers are:

```text
SoftmaxCrossEntropyHVPFloat32Kernel
SoftmaxCrossEntropyHVPFloat16Kernel
SoftmaxCrossEntropyHVPBFloat16Kernel
SoftmaxCrossEntropyHVPCachedFloat32Kernel
SoftmaxCrossEntropyHVPCachedFloat16Kernel
SoftmaxCrossEntropyHVPCachedBFloat16Kernel
```

Every registered task fixes `B=4096`, `C=1000`. Constructors also accept
`batch_size`, `num_classes` and `autotune` for direct tests and baseline
sweeps. The kernel supports `B > 0` and `1 <= C <= 8192`; booleans are not
accepted as dimensions. All inputs are contiguous 2D tensors on the same
CUDA device. Each class enforces its vector dtype and fixed instance shape.

The inputs must be finite. Cached probabilities must be nonnegative,
normalized FP32 softmax probabilities, up to ordinary FP32 roundoff. Cached
mode neither computes a softmax nor renormalizes its input. Value-domain
checks do not run on every launch, to avoid GPU synchronization in the timed
wrapper; callers are responsible for these preconditions.

FP16/BF16 inputs are promoted **before** exponentials, multiplication,
subtraction or reductions. Lower-precision inputs still quantize the original
logits/direction: FP32 accumulation cannot recover information lost at input
conversion. Cached probabilities remain FP32 to avoid an additional
low-precision probability approximation. To compare the two variants on the
same data, form cached `p = logits.float().softmax(-1)` from the fused variant's
already-quantized logits, and reuse the same vector.

## 3. The Triton implementation and launch contract

The wrappers and static kernels are in
[softmax_ce_hvp.py](../ptx_gym/kernels/common/softmax_ce_hvp.py). Each CTA owns
one entire row; the grid is `(B,)`. The pointer/scalar order is
`(output_ptr, input_ptr, vector_ptr, n_cols)`. `BLOCK_SIZE` is a constexpr
equal to the next power of two at least `C`. At the canonical shape it is
1,024. Ordinary `n_cols` remains a runtime argument through the Gym's
`jit_fixed_parameters` helper. Use the extracted PTX signature, including any
compiler-added parameters, as the final ABI authority.

The fused kernel performs:

1. Masked loads of logits and direction; logits in padded lanes become
   negative infinity and direction values become zero.
2. FP32 row maximum and `exp(z - max(z))`. The subtraction prevents overflow
   for large finite logits; at least one valid lane has exponential one.
3. A denominator reduction and FP32 probabilities.
4. A reduction of `p * v`, followed by masked stores of `p * (v - s)`.

The cached kernel loads FP32 probabilities with zero in padded lanes, computes
the projection reduction, and stores the action. Intermediate probabilities
and reductions stay inside the kernel rather than being written to global
memory. Register allocation and any spills are compiler/device-dependent.

Both wrappers expose `num_warps=(1,2,4,8,16)` for tuning, start at four warps,
and use `num_stages=1`. The shared Gym base class selects the fastest valid
configuration on the actual device. A shape change changes the constexpr
block width and can change the best warp count; retune for each shape/device.
PTX replacements use the wrapper's existing `ptx_launch_kwargs` path.

Both `autotune_tolerance` and `verification_tolerance` are `1e-6`. This is
tighter than the Gym's generic default because many HVP entries are small.
Dedicated tests also check the reference against independent mathematical
oracles and ensure a zero-writing candidate does not pass on nontrivial data.

## 4. Why keep a cached variant for CG?

At a fixed model parameter point, a generalized Gauss–Newton product is

\[
(G+\lambda I)d=J^T H_z(Jd)+\lambda d.
\]

An inner CG solve changes the direction `d` repeatedly while keeping the
linear operator fixed. Compute softmax probabilities once at that parameter
point, then reuse the cached HVP with directions `v=Jd`. Refresh the
probabilities when the outer optimization point changes.

This kernel implements only `H_z v`. The JVP `Jd`, VJP `J^T(...)`, damping,
CG iterations and parameter update are separate operations. Accelerating this
kernel does not imply the same speedup for the full optimizer. Fused and
cached timings answer different questions: cached timing excludes the cost
of preparing probabilities.

## 5. What can be validated without a GPU?

[ptx_gym.reference](../ptx_gym/reference) requires Torch and does not import
Triton. It exports:

```python
from ptx_gym.reference import softmax_ce_hvp, softmax_ce_hvp_from_probs

# Both inputs can live on CPU. The output uses FP32.
output = softmax_ce_hvp(logits, vector)
output_cached = softmax_ce_hvp_from_probs(logits.float().softmax(-1), vector)
```

[CPU tests](../tests/cpu/test_softmax_ce_hvp.py) compare with float64 second
derivatives of actual Torch cross-entropy, and with small explicit Hessians.
They also exercise label independence, batch-mean scaling, zero vectors,
`H*1≈0`, shift invariance, zero row sums, linearity, symmetry and
`v^T H v >= 0`. Shape/dtype/layout errors and extreme finite logits are tested.
Large cases use an autograd HVP, not a dense 8,192-square Hessian.

[GPU tests](../tests/gpu/test_softmax_ce_hvp_gpu.py) cover all six wrappers,
tail masks, finite outputs, input preservation, metadata/registration and
reinjection of PTX into the same operator. CPU success establishes the math
and Python-side contracts; it does not establish Triton compilation,
sanitizer cleanliness or device performance. Those remain remote validation
steps until actual reports are collected.
