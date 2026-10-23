"""Volta primitive correctness, tails, duplicate indices and CUDA graph replay."""
import os

import pytest
import torch

from fairchem.core.models.uma.uma_nitro import _inference as inference
from fairchem.core.models.uma.uma_nitro._volta import gather, scatter


@pytest.mark.parametrize('edges,channels', [(0, 7), (1, 7), (17, 8), (513, 128)])
@pytest.mark.parametrize('strided', [False, True])
def test_gather_scatter(edges, channels, strided):
    assert os.environ.get('SLURM_JOB_ID')
    torch.manual_seed(11)
    step = 2 if strided else 1
    src = torch.randn(13, 9, channels * step, device='cuda')[:, :, ::step]
    index = torch.randint(0, 13, (edges * step,), dtype=torch.int32, device='cuda')[::step]
    out = torch.empty(edges, 9, channels * step, device='cuda')[:, :, ::step]
    gather(src, index, out)
    torch.testing.assert_close(out, src[index.long()], atol=0, rtol=0)
    grad = torch.randn_like(out).half()
    scale = torch.randn(edges * step, device='cuda')[::step]
    target = torch.randn_like(src)
    expected = target.clone().index_add_(0, index, grad.float() * scale[:, None, None])
    scatter(grad, index, scale, target)
    torch.testing.assert_close(target, expected, atol=2e-5, rtol=2e-5)
    if edges:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gather(src, index, out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather(src, index, out)
        src.add_(1)
        graph.replay()
        torch.testing.assert_close(out, src[index.long()], atol=0, rtol=0)


def test_volta_device_opt_in(monkeypatch):
    monkeypatch.setattr(inference, '_CUDA_CAPABILITY', (7, 0))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *_: (7, 0))
    monkeypatch.delenv('UMA_NITRO_ENABLE_VOLTA', raising=False)
    with pytest.raises(inference.UMANitroNotSupportedError):
        inference._validate_device('cuda')
    monkeypatch.setenv('UMA_NITRO_ENABLE_VOLTA', '1')
    inference._validate_device('cuda')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *_: (7, 5))
    with pytest.raises(inference.UMANitroNotSupportedError):
        inference._validate_device('cuda')
    policy = inference._kernel_policy((7, 0))
    assert not any((policy.use_tf32, policy.use_cute_kernels,
                    policy.use_hopper_gemm, policy.supports_vector_atomics))
