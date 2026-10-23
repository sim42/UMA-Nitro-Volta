"""Run inside a Slurm GPU allocation before enabling experimental Nitro."""
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + i, tl.load(x + i, i < n, 0) + 1, i < n)


@triton.jit
def dot_kernel(a, b, c, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(a + i[:, None] * BLOCK + i[None, :])
    y = tl.load(b + i[:, None] * BLOCK + i[None, :])
    z = tl.dot(x, y, input_precision="ieee")
    tl.store(c + i[:, None] * BLOCK + i[None, :], z)


assert os.environ.get("SLURM_JOB_ID"), "Use srun --gres=gpu:v100:1"
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "triton": triton.__version__, "gpu": torch.cuda.get_device_name(),
                  "capability": torch.cuda.get_device_capability(),
                  "job_id": os.environ["SLURM_JOB_ID"]}), flush=True)
x = torch.randn(513, device="cuda")
y = torch.empty_like(x)
add_kernel[(3,)](x, y, x.numel(), 256)
torch.testing.assert_close(y, x + 1)
for dtype in (torch.float32, torch.float16):
    a = torch.randn(32, 32, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    c = torch.empty(32, 32, device="cuda")
    kernel = dot_kernel[(1,)](a, b, c, 32, num_warps=4, num_stages=1)
    torch.testing.assert_close(c, a.float() @ b.float(), atol=1e-4, rtol=1e-4)
    print(dtype, "dot passed", "mma.sync" in kernel.asm["ptx"], flush=True)
print("PROBE PASS", flush=True)
