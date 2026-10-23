# UMA-Nitro-Volta

An experimental NVIDIA V100 (`sm_70`) backend for Meta's UMA Nitro, supporting
single-system energy, force and stress prediction with `uma-s-1p2p1`.
Based on [FAIR Chemistry](https://github.com/facebookresearch/fairchem).

The backend uses Volta-compatible Triton kernels and cuBLAS while retaining
Nitro's CUDA Graph execution. It requires explicit opt-in and does not use
Hopper-only instructions or TF32. Model weights are unchanged.

## Install

Requirements: Linux, Python 3.11, an NVIDIA V100 and a CUDA 12-compatible driver.
Use the CUDA 12.6 PyTorch build; CUDA 13 builds do not support Volta.

```bash
git clone https://github.com/sim42/UMA-Nitro-Volta.git
cd UMA-Nitro-Volta
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-volta.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export UMA_NITRO_ENABLE_VOLTA=1
```

Obtain `uma-s-1p2p1.pt` from the [official model repository](https://huggingface.co/facebook/UMA)
after accepting its licence. Weights are not included here.

## Use with ASE

```python
from ase.build import molecule
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import inference_settings_nitro

settings = inference_settings_nitro()
settings.compile = False
predictor = MLIPPredictUnit("/path/to/uma-s-1p2p1.pt", "cuda", inference_settings=settings)
atoms = molecule("H2O")
atoms.info.update(charge=0, spin=1)
atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())           # eV/Angstrom
```

Use a separate predictor when composition, task, charge or spin changes. This
backend supports single-system inference; training and higher-order derivatives
are outside its validated scope.

## Performance

A 1,000-atom silicon ASE benchmark on one V100 16 GB averaged **97.16 ms/step**
after warmup. This corresponds to **0.889 ns/day at a normalized 1 fs timestep**;
the actual benchmark integration timestep was 0.1 fs. Initial loading and
compilation are excluded. Performance varies with system size and connectivity.

## Licence

Code retains the upstream MIT licence and applicable third-party notices.
Model weights are covered separately by the
[FAIR Chemistry model licence](https://huggingface.co/facebook/UMA).
