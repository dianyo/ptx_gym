# Run the HVP benchmarks locally and on DGX Spark

Read the [architecture guide](architecture.md) and [benchmark definition](softmax_ce_hvp.md)
for the implementation and numerical contract. These instructions target Linux
ARM64 DGX Spark/GB10. Device code is compiled on the remote machine; no GPU
build is required on the Mac.

**Validation status:** the contribution was developed on macOS without CUDA.
CPU tests can be run there. GPU compilation, numerical tests, sanitizers,
latencies and profiles must be validated by the remote commands below. No
remote performance claim is included.

## 1. Local CPU development

Run commands from the cloned **ptx_gym repository root**, not its parent:

```bash
cd /Users/dianyo/Desktop/UCB/triton_ai_compiler/ptx_gym
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pytest --import-mode=importlib tests/cpu \
  ptx_gym/helpers/tests/test_environment.py \
  ptx_gym/evaluation/tests/test_types.py -q
python -m ptx_gym.benchmarks.softmax_ce_hvp --help
```

The environment can use Python 3.10–3.14 with a compatible Torch wheel. Local
development here used Python 3.13/Torch 2.11.0. The explicit pytest import mode
avoids a name collision between the repository-root compatibility package and
the nested source package when collecting different test directories together.
Do not install the root project in this CPU environment: its installer builds
the native patched Triton distribution.

## 2. Transfer the five commits

The feature branch is `feat/softmax-ce-hvp`. Choose either transfer route;
neither environments nor generated GPU artifacts belong in Git.

### Push and clone

On the Mac, when you want the branch on your fork:

```bash
cd /Users/dianyo/Desktop/UCB/triton_ai_compiler/ptx_gym
git push -u origin feat/softmax-ce-hvp
```

On Spark:

```bash
mkdir -p ~/research
cd ~/research
git clone --recurse-submodules --branch feat/softmax-ce-hvp \
  https://github.com/dianyo/ptx_gym.git
cd ptx_gym
git rev-parse HEAD
```

For an existing clean remote clone, use `git fetch origin`,
`git switch feat/softmax-ce-hvp`, then `git pull --ff-only`. Preserve any remote
local work before switching branches.

### Git bundle, without pushing

Replace `YOUR_SPARK_HOST` with your SSH host or configured alias. On the Mac:

```bash
cd /Users/dianyo/Desktop/UCB/triton_ai_compiler/ptx_gym
git bundle create /tmp/softmax-ce-hvp.bundle feat/softmax-ce-hvp
git bundle verify /tmp/softmax-ce-hvp.bundle
scp /tmp/softmax-ce-hvp.bundle YOUR_SPARK_HOST:~/softmax-ce-hvp.bundle
```

On Spark:

```bash
mkdir -p ~/research
cd ~/research
git clone --branch feat/softmax-ce-hvp ~/softmax-ce-hvp.bundle ptx_gym
cd ptx_gym
git remote set-url origin https://github.com/dianyo/ptx_gym.git
git submodule update --init --recursive
git rev-parse HEAD
```

The bundle includes the feature branch's reachable history. Setting `origin`
does not publish the branch. Compare `git rev-parse HEAD` on both machines to
confirm you are testing the same source revision.

## 3. Prepare the remote GPU environment

Start in `~/research/ptx_gym` (or your chosen remote clone location). DGX Spark
uses ARM64 and has a distinct CUDA target from a B200. NVIDIA's
[Spark build guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html)
documents native compilation for capability 12.1. Confirm what your machine
actually reports:

```bash
uname -m
nvidia-smi
nvcc --version
command -v uv
command -v compute-sanitizer
command -v ncu
```

Use an updated Spark software stack with working NVIDIA drivers and CUDA
tools. `uv`, a C/C++ build toolchain, Python development headers, CMake and Ninja
must be available; uv installs the Python build dependencies in isolation.
Python 3.12 is a practical default for the remote environment.

Create an isolated environment, install CUDA-enabled ARM64 Torch, then build
this repository's patched Triton:

```bash
uv venv --python 3.12 .venv-gpu
source .venv-gpu/bin/activate
uv pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu130
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'Install a CUDA-enabled Torch build before continuing'
print('GPU:', torch.cuda.get_device_name())
print('Capability:', torch.cuda.get_device_capability())
assert torch.cuda.get_device_capability() == (12, 1), 'Check that this is the intended GB10 device'
PY
MAX_JOBS=8 uv pip install -e . -v
```

The [official CUDA 13.0 Torch wheel index](https://download.pytorch.org/whl/cu130/torch/)
contains the 2.11.0 ARM64 Python 3.12 wheel. This is a specified starting
configuration, not a claim that this exact combined stack has been tested on
your remote hardware. The PyTorch runtime CUDA version and the compiler's
bundled assembler version need not be identical. Record both in experiment
artifacts. The initial native build downloads dependencies and can take time;
lower `MAX_JOBS` if host memory pressure is high.

The root uv configuration prevents Torch from bringing in stock Triton. Do
not install another Triton wheel after building the patched distribution.
Validate the actual imported package and injection capability:

```bash
python - <<'PY'
import inspect
import torch
import triton
from ptx_gym.helpers.environment import get_ptx_system_config, get_ptxas_path
print('Triton:', triton.__version__, triton.__file__)
assert 'ptx' in inspect.signature(triton.jit).parameters, 'Patched Triton is not imported'
version, target, address_size = get_ptx_system_config()
print('PTX ISA / target / bits:', version, target, address_size)
print('Gym assembler:', get_ptxas_path(target))
print('GPU:', torch.cuda.get_device_name())
PY
```

This fork's bundled Triton identifies as 3.6.0 and ships a dedicated Blackwell
assembler configured at CUDA 12.9.86. That maps to PTX 8.8. The backend normally
emits `sm_121a` for capability 12.1. Let the compiler detect the target and
preserve the emitted `.version`, `.target` and entry ABI when replacing PTX.
The runner also checks that Gym and Triton resolve the same assembler.

## 4. Run focused correctness tests, then smoke checks

```bash
export TRITON_CACHE_DIR="$PWD/.cache/softmax-ce-hvp-gb10"
python -m pytest --import-mode=importlib tests/cpu tests/gpu \
  ptx_gym/helpers/tests/test_environment.py \
  ptx_gym/evaluation/tests/test_types.py -q
python -m ptx_gym.benchmarks.softmax_ce_hvp --mode smoke
```

The GPU test module must execute rather than be skipped. It covers all six
variants, including widths 1, 2, 31, 32, 33, 255, 256, 257, 1,000, 1,024,
1,025, 4,096 and 8,192. The smoke runner uses each canonical `[4096,1000]`
task with four warps and no autotuning. It compares against Torch, assembles
the extracted PTX, reinjects it into the same operator, and checks finite FP32
outputs and input preservation across three random trials.

Smoke mode does not run sanitizers, profiling or the long Gym timing protocol.
It exits nonzero on a setup/task failure. A CPU-only invocation records a setup
failure and no GPU results. To select one task during debugging:

```bash
python -m ptx_gym.benchmarks.softmax_ce_hvp --mode smoke \
  --kernel SoftmaxCrossEntropyHVPFloat32Kernel
```

## 5. Full self-PTX evaluation, sanitizers and profiling

Run the full mode after correctness and smoke checks pass. Keep other GPU
workloads idle during timing.

```bash
export PTX_MEMORY_SANITIZER="$(command -v compute-sanitizer)"
export NCU_PATH="$(command -v ncu)"
test -x "$PTX_MEMORY_SANITIZER"
test -x "$NCU_PATH"
python -m ptx_gym.benchmarks.softmax_ce_hvp --mode full \
  --kernel SoftmaxCrossEntropyHVPFloat32Kernel
```

Omit `--kernel` to evaluate all six, or repeat it to select several:

```bash
python -m ptx_gym.benchmarks.softmax_ce_hvp --mode full
```

Full mode uses `get_kernel_data`, `dump_kernel_ptx` and `evaluate_candidate`
in one process, so each task's metadata, PTX and evaluator share one tuned
baseline. It runs the normal Gym protocol: assembly, four Compute Sanitizer
tools, 1,000 randomized PTX-versus-Triton trials, 50 alternating timing rounds,
and Nsight Compute. Each task may take minutes depending on the machine/tools.

The candidate is **Triton-generated reference PTX**, not an optimized candidate.
Expect noisy speedup around 1×; there is no exact-1× pass condition. The
runner's full `passed` also requires a successful, nonempty NCU metrics report.
Gym's own `passed` only covers assembly/correctness, so inspect both
`gym_passed` and `full_checks_complete` if profiling fails.

Volta is disabled and these HVP tasks have no Volta specifications.

## 6. Sweep baseline shapes

```bash
python -m ptx_gym.benchmarks.softmax_ce_hvp --mode sweep
```

The default sweep covers `(B,C)` values `(1,10)`, `(32,256)`, `(256,1000)`,
`(4096,1000)`, `(1024,4096)` and `(256,8192)` for all six variants. For a
smaller selected sweep:

```bash
python -m ptx_gym.benchmarks.softmax_ce_hvp --mode sweep \
  --kernel SoftmaxCrossEntropyHVPFloat32Kernel \
  --kernel SoftmaxCrossEntropyHVPCachedFloat32Kernel \
  --shape 32x256 --shape 256x1000 --shape 256x8192
```

Each shape gets its own tuned Triton instance, correctness comparison,
reference PTX, launch metadata and Torch/Triton baseline latency percentiles.
This short sweep uses 100 ms warmup and 200 ms sampling, rather than the full
Gym candidate protocol. It does not evaluate custom PTX, run tools or emit a
candidate speedup. `--shape` is deliberately rejected in smoke/full modes:
sanitizer/profiler children reconstruct the registered canonical task.

## 7. Read and retain artifacts

Every invocation prints a fresh timestamped directory under
`results/softmax_ce_hvp/`; use `--output-dir /path/to/experiment` to change its
parent. Existing runs are not overwritten.

```text
<timestamp>-<mode>/
  environment.json          # Git/software/device/assembler/cache information
  summary.json              # Overall verdict and individual task reports
  <kernel-id>[-<shape>]/
    metadata.json           # Shape, precision, tuning and available contract/resources
    reference.ptx
    compilation.json        # Smoke assembly diagnostics
    gym_result.json         # Full Gym verdict, timings, sanitizer/NCU reports
    report.json             # Mode-specific result or error
```

Files produced before a failure are retained; missing files mean that stage
was not reached. The full public API retains JSON NCU metrics but removes the
source CSV field; the Gym also cleans its temporary raw `.ncu-rep` file. These
artifacts therefore contain the JSON profile, not a reusable Nsight GUI
session. Preserve the entire run directory when comparing experiments. The
`git_dirty` field records tracked source changes; commit research changes
before collecting final measurements.

For later B200 runs, rebuild and retune on that machine, retain its detected
target, and use separate results/cache directories. Compare per-device
candidate speedups only after collecting actual optimized candidate results.

## 8. Troubleshooting

| Symptom | Action |
| --- | --- |
| `CUDA is unavailable` | Check `torch.version.cuda`, device access and the ARM64 CUDA-enabled Torch installation; CPU Torch cannot execute this suite. |
| Missing `ptx` parameter on `triton.jit` | Activate the intended environment and rebuild the root distribution; inspect `triton.__file__`. |
| GPU test module skipped | Install the patched build and verify CUDA before interpreting the test summary as GPU validation. |
| Unsupported `sm_121a` or PTX ISA mismatch | Inspect the recorded target and selected `ptxas-blackwell --version`; rebuild the bundled tools. Keep Gym and Triton's assembler selection aligned. |
| `nvidia-smi` memory fields show N/A | Expected for Spark UMA; the helper falls back to CUDA's conservative free-memory estimate. See [NVIDIA's known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html). |
| `ERR_NVGPUCTRPERM` / empty NCU profile | Ask the remote administrator to enable access to NVIDIA profiling counters; retain the failed report and rerun full mode after access is available. |
| Sanitizer/NCU timeout or unsupported tool | Check tool versions against the remote driver/GPU and read the individual diagnostics in `gym_result.json`; the run is incomplete until those checks pass. |
| Build stops under host memory pressure | Reduce `MAX_JOBS`; the native Triton/LLVM build has a separate memory footprint from the benchmark tensors. |
| Noisy or unexpected self-PTX speedup | Stop competing GPU jobs, check thermal/power conditions, and compare repeated complete runs on the same device/toolchain. |

Evaluation deletes the configured Triton cache directory. Keep
`TRITON_CACHE_DIR` dedicated to this experiment, and use a fresh directory
after a toolchain/device change.
