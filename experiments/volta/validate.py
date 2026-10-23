"""Compare real checkpoint Nitro outputs with the uncompiled FP32 reference.

Run through run-slurm.sh. No implicit download, CPU fallback or torch.compile.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.uma_nitro import _inference as nitro
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, inference_settings_nitro


def batch(atoms, task, charge=0, spin=1):
    atoms = atoms.copy()
    atoms.info.update(charge=charge, spin=spin)
    data = AtomicData.from_ase(atoms, task_name=task, r_edges=False,
                               r_data_keys=["charge", "spin"])
    return data_list_collater([data], otf_graph=True)


def predict(model, data):
    out = model.predict(data.clone())
    return {k: v.detach().cpu().double() for k, v in out.items()}


def check_nitro(model):
    assert model.inference_settings.nitro
    assert type(model.model.module).__name__ == "UMANitroModel", "Silent fallback"


def elapsed(model, data, repeats):
    model.predict(data.clone())
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(data.clone())
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="experiments/volta/results/validation.json")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--case", default="all")
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "GPU work must run inside Slurm"
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    report = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "triton": nitro.triton.__version__, "backend": nitro.kernel_backend_summary(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "cuda_graph": nitro._USE_GRAPH, "cases": []}
    systems = [
        ("water", molecule("H2O"), "omol", 0, 1),
        ("ethanol", molecule("CH3CH2OH"), "omol", 0, 1),
        ("water_ion", molecule("H2O"), "omol", 1, 2),
        ("hydrogen", Atoms("H", positions=[[0, 0, 0]]), "omol", 0, 2),
        ("no_edges", Atoms("O2", positions=[[0, 0, 0], [0, 0, 20]]), "omol", 0, 3),
        ("copper32", bulk("Cu", "fcc", cubic=True) * (2, 2, 2), "omat", 0, 0),
        ("copper108", bulk("Cu", "fcc", cubic=True) * (3, 3, 3), "omat", 0, 0),
    ]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = MLIPPredictUnit(args.checkpoint, "cuda", inference_settings=InferenceSettings(
        merge_mole=False, compile=False, execution_mode="general", activation_checkpointing=False))
    for name, atoms, task, charge, spin in systems:
        if args.case not in ("all", name):
            continue
        print("CASE", name, flush=True)
        data = batch(atoms, task, charge, spin)
        expected = predict(reference, data)
        settings = inference_settings_nitro()
        settings.compile = False
        fast = MLIPPredictUnit(args.checkpoint, "cuda", inference_settings=settings)
        actual = predict(fast, data)
        check_nitro(fast)
        assert actual.keys() == expected.keys()
        errors = {}
        for key in actual:
            assert torch.isfinite(actual[key]).all(), key
            errors[key] = float((actual[key] - expected[key]).abs().max())
        assert errors["energy"] / len(atoms) < 0.005, errors
        assert errors["forces"] < 0.05 + 0.01 * float(expected["forces"].abs().max()), errors
        if "stress" in errors:
            assert errors["stress"] < 0.005 + 0.01 * float(expected["stress"].abs().max()), errors
        # Exercise replay with changed internal coordinates, not just translation.
        moved = atoms.copy()
        moved.positions[0, 0] += 0.01
        moved_data = batch(moved, task, charge, spin)
        replay = predict(fast, moved_data)
        moved_ref = predict(reference, moved_data)
        assert (replay["forces"] - moved_ref["forces"]).abs().max() < 0.06 + 0.01 * moved_ref["forces"].abs().max()
        check_nitro(fast)
        row = {"name": name, "atoms": len(atoms), "task": task,
               "energy_error_ev_per_atom": errors["energy"] / len(atoms),
               "max_abs_error": errors, "nitro_ms": elapsed(fast, data, args.repeats),
               "reference_ms": elapsed(reference, data, args.repeats)}
        if name == "water":
            delta = 0.002
            energies = []
            for sign in (-1, 1):
                displaced = atoms.copy()
                displaced.positions[1, 2] += sign * delta
                energies.append(predict(fast, batch(displaced, task, charge, spin))["energy"].item())
            fd_force = -(energies[1] - energies[0]) / (2 * delta)
            row["finite_difference_force_error_ev_a"] = abs(fd_force - actual["forces"][1, 2].item())
            assert row["finite_difference_force_error_ev_a"] < 0.05, row
        report["cases"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        nitro._flush_graphs()
        del fast
        gc.collect()
        torch.cuda.empty_cache()
    assert report["cases"], "No matching case"
    report["passed"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("VALIDATION PASS", flush=True)


if __name__ == "__main__":
    main()
