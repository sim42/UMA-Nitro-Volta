# Volta experiment

## Validation

`probe.py` checks a real V100 allocation with scalar addition, FP32 dot and FP16
dot, and reports whether the generated PTX contains `mma.sync`. With Triton 3.8
these small dot probes use SIMT. Tensor Core use by cuBLAS is a separate path.
`compile_probe.py` can check compilation for sm_70 without allocating a GPU.

`test_kernels.py` checks fused gather/scatter against PyTorch, including empty
input, non-power-of-two tails, repeated indices, strided storage and graph replay.
Run it inside a Slurm allocation with the environment in `run-slurm.sh`:

```bash
.venv/bin/python -m pytest -q experiments/volta/test_kernels.py
```

`validate.py` uses the real checkpoint on water, ethanol, a charged doublet,
isolated hydrogen, separated oxygen atoms, and periodic copper (32 and 108
atoms). It checks perturbed geometry replay and finite-difference force agreement.
Absolute limits follow upstream Nitro's reduced-precision acceptance tests:
energy < 5 meV/atom; max force error < 0.05 eV/Å + 1% of the largest reference
component; stress < 0.005 eV/Å³ + 1%. Finite-difference water force tolerance is
0.05 eV/Å at displacement 0.002 Å. These are regression tolerances, not claims of
chemical accuracy or validation for long MD trajectories.

`--case water` selects a short smoke test; `--repeats 30` increases timing samples.
`UMA_NITRO_NO_CUDAGRAPH=1` exercises the eager Nitro path for diagnosis.

## WebMolecule integration boundary

Use this fork in a separate compute environment, inside the existing Slurm GPU
worker. Set `PYTHONPATH` to `packages/fairchem-core/src` and set
`UMA_NITRO_ENABLE_VOLTA=1` before importing FairChem. Load the local checkpoint
with `MLIPPredictUnit(path, "cuda", inference_settings=...)`, using
`inference_settings_nitro()` with `compile=False`. Wrap it with
`FAIRChemCalculator(predictor, task_name="omol")`; set `atoms.info["charge"]` and
`atoms.info["spin"]` explicitly (spin multiplicity, not number of unpaired electrons).

The ASE calculator returns energy in eV and forces in eV/Å. An engine that expects
energy and gradient in kcal/mol must convert by its own eV→kcal/mol constant and
return the **negative** force as gradient. Keep one prepared predictor per fixed
composition, charge, spin and task: Nitro merges experts on first use, and a
changed system otherwise triggers a fallback. Do not share a prepared predictor
across arbitrary classroom molecules. Serialize access to a GPU worker, since
Nitro uses module-global CUDA graph caches.

This experiment does not deploy the public WebMolecule service or modify its
private source. Full application acceptance and long MD validation remain separate
from the kernel and model checks here.
