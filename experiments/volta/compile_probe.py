"""Compile a small FP32 Triton kernel for sm_70 without taking a GPU."""
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def dot(a, b, c, B: tl.constexpr):
    i = tl.arange(0, B)
    x = tl.load(a + i[:, None] * B + i[None, :])
    y = tl.load(b + i[:, None] * B + i[None, :])
    z = tl.dot(x, y, input_precision="ieee")
    tl.store(c + i[:, None] * B + i[None, :], z)

for dtype in ('fp32', 'fp16'):
    try:
        k = triton.compile(ASTSource(dot, signature={'a': '*'+dtype, 'b': '*'+dtype, 'c': '*fp32', 'B': 'constexpr'}, constexprs={'B': 32}), target=GPUTarget('cuda',70,32), options={'num_warps':4, 'num_stages':1})
        print(dtype, 'COMPILE PASS', 'mma.sync' in k.asm['ptx'], flush=True)
    except Exception as exc:
        print(dtype, type(exc).__name__, str(exc), flush=True)
        raise
