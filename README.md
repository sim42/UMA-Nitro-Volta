# UMA-Nitro-Volta (experimental)

An independent NVIDIA V100 (`sm_70`) port of Meta's UMA Nitro, intended for
ASE-based molecular modelling and WebMolecule compute workers. This is not an
official Meta release. Work is isolated on branch `UMA-Nitro-Volta`.

Based on [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem)
commit `645af7d03e91138ef07e51de387316c9e2e152dc` (2026-10-22).
The original repository and licence notices are retained; see
[upstream README](experiments/volta/UPSTREAM_README.md).

## Architecture

- Explicit `UMA_NITRO_ENABLE_VOLTA=1` opt-in; only compute capability **7.0** is
  admitted. Other unvalidated architectures keep upstream checks.
- Hopper CuTe/WGMMA GEMM is replaced by upstream's cuBLAS implementation.
  FP16 inputs use FP32 accumulation. Float32 matmul uses IEEE precision.
- TF32 rounding and Hopper vector atomics remain disabled on V100.
- Volta-specific fused Triton gather/scatter handles strided tensors, converts
  FP16 gradients, rescales and accumulates with scalar FP32 atomics. Pointer
  arithmetic is 64 bit. No CuTe import is needed on V100.
- The remaining Nitro fused kernels, conservative force differentiation,
  checkpoint checksum gate, finite-output/capacity guards and CUDA graph
  orchestration are retained.
- Nitro calls Triton directly; this does not enable `torch.compile` on V100.

## Reproduce

Use Linux, Python 3.11, a CUDA 12 compatible NVIDIA driver and a V100.
Install outside a GPU allocation:

```bash
bash experiments/volta/setup.sh
mkdir -p checkpoints
# Supply your own authorised checkpoint; this repository contains no model weights.
# checkpoints/uma-s-1p2p1.pt
sha256sum checkpoints/uma-s-1p2p1.pt
sbatch experiments/volta/run-slurm.sh
```

Expected checkpoint SHA-256:
`b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce`.
Weights retain their separate [FAIR Chemistry model licence](https://huggingface.co/facebook/UMA).
Code is distributed under the original MIT licence; the copied NVIDIA CuTe
source retains its BSD notices even though this path does not load it.

The Slurm script runs a real GPU Triton probe, then compares the actual
`UMANitroModel` against the uncompiled, unmerged FP32 reference. It rejects silent
fallbacks. Results include warm median wall latency, error metrics and the Slurm
job ID. Initial compilation and CUDA graph capture are excluded from warm timing.
See [validation details](experiments/volta/README.md).

Public GitHub Actions are disabled. GPU work is submitted manually through Slurm;
this repository is not attached to a privileged self-hosted runner.
