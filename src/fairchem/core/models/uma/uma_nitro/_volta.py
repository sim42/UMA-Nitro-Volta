"""Volta gather/scatter kernels using scalar FP32 atomics and 64-bit offsets.

Copyright (c) 2026 UMA-Nitro-Volta contributors. MIT license.
No CuTe, WGMMA, TF32, BF16 or asynchronous shared-memory copy instructions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rows_kernel(
    Src,
    Index,
    Scale,
    Out,
    E: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    IS: tl.constexpr,
    CS: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    edge = offset // (S * C)
    sph = (offset // C) % S
    channel = offset % C
    valid = edge < E
    node = tl.load(Index + edge * IS, valid, 0).to(tl.int64)
    if SCATTER:
        x = tl.load(Src + edge * SS0 + sph * SS1 + channel * SS2, valid, 0).to(
            tl.float32
        )
        scale = tl.load(Scale + edge * CS, valid, 0)
        tl.atomic_add(
            Out + node * OS0 + sph * OS1 + channel * OS2,
            x * scale,
            valid,
            sem="relaxed",
        )
    else:
        x = tl.load(Src + node * SS0 + sph * SS1 + channel * SS2, valid, 0)
        tl.store(Out + edge * OS0 + sph * OS1 + channel * OS2, x, valid)


def gather(src: torch.Tensor, index: torch.Tensor, out: torch.Tensor) -> None:
    """Gather (N,S,C) node rows into a possibly strided (E,S,C) buffer."""
    assert src.ndim == out.ndim == 3
    assert out.shape == (index.numel(), *src.shape[1:])
    if out.numel() == 0:
        return
    _rows_kernel[(triton.cdiv(out.numel(), 256),)](
        src,
        index,
        index,
        out,
        index.numel(),
        *src.shape[1:],
        *src.stride(),
        *out.stride(),
        index.stride(0),
        0,
        False,
        256,
        num_warps=4,
        num_stages=1,
    )


def scatter(
    grad_edge: torch.Tensor, index: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> None:
    """Fuse fp16 conversion, per-edge scaling and scalar fp32 scatter-add."""
    assert grad_edge.ndim == out.ndim == 3
    assert grad_edge.shape == (index.numel(), *out.shape[1:])
    assert scale.numel() == index.numel()
    assert out.dtype == torch.float32
    if grad_edge.numel() == 0:
        return
    _rows_kernel[(triton.cdiv(grad_edge.numel(), 256),)](
        grad_edge,
        index,
        scale,
        out,
        index.numel(),
        *out.shape[1:],
        *grad_edge.stride(),
        *out.stride(),
        index.stride(0),
        scale.stride(0),
        True,
        256,
        num_warps=4,
        num_stages=1,
    )
