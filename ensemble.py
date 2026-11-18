import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


STEP_RE = re.compile(r"^(?P<var>.+)_step(?P<step>\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weighted-average saved prediction files from multiple output/test experiments."
    )
    parser.add_argument(
        "--root",
        default="./output/test",
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment folder names under --root, or explicit paths.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for experiments. If omitted, equal weights are used.",
    )
    parser.add_argument(
        "--vars",
        "--save_vars",
        dest="variables",
        nargs="+",
        required=True,
        help="Variables to ensemble, for example: tos thetao.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Optional prediction steps to ensemble. If omitted, all steps found in the first experiment are used.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write ensembled predictions.",
    )
    parser.add_argument(
        "--allow_mismatched_init_times",
        action="store_true",
        help="Allow averaging experiments whose init_times.json differ. Use only when you know sample order matches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Compatibility flag. Existing output files are overwritten by default.",
    )
    return parser.parse_args()


def resolve_experiment_paths(root: str, experiments: Sequence[str]) -> List[str]:
    paths = []
    for exp in experiments:
        path = exp if os.path.isdir(exp) else os.path.join(root, exp)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Experiment directory not found: {path}")
        preds_dir = os.path.join(path, "preds")
        if not os.path.isdir(preds_dir):
            raise FileNotFoundError(f"Missing preds directory: {preds_dir}")
        paths.append(path)
    return paths


def normalize_weights(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite numbers.")
    weight_sum = weights.sum()
    if weight_sum == 0:
        raise ValueError("Weights sum to zero.")
    return weights / weight_sum


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def check_init_times(exp_paths: Sequence[str], allow_mismatch: bool) -> Optional[List[str]]:
    init_times_path = os.path.join(exp_paths[0], "init_times.json")
    if not os.path.exists(init_times_path):
        return None

    init_times = read_json(init_times_path)
    if allow_mismatch:
        return init_times

    for exp_path in exp_paths[1:]:
        other_path = os.path.join(exp_path, "init_times.json")
        if not os.path.exists(other_path):
            raise FileNotFoundError(f"Missing init_times.json: {other_path}")
        other = read_json(other_path)
        if other != init_times:
            raise ValueError(
                "init_times.json differs across experiments. "
                "Averaging would be invalid unless sample order is known to match."
            )
    return init_times


def list_prediction_files(preds_dir: str, variable: str) -> Tuple[Optional[str], Dict[int, str]]:
    single_file = os.path.join(preds_dir, f"{variable}.pt")
    single = single_file if os.path.exists(single_file) else None
    steps = {}
    for name in os.listdir(preds_dir):
        match = STEP_RE.match(name)
        if match and match.group("var") == variable:
            steps[int(match.group("step"))] = os.path.join(preds_dir, name)
    return single, steps


def resolve_variable_files(
    exp_paths: Sequence[str],
    variable: str,
    requested_steps: Optional[Sequence[int]],
) -> List[Tuple[str, List[str]]]:
    first_preds_dir = os.path.join(exp_paths[0], "preds")
    first_single, first_steps = list_prediction_files(first_preds_dir, variable)

    if requested_steps is None and first_single is not None and len(first_steps) == 0:
        expected_names = [f"{variable}.pt"]
    else:
        if requested_steps is None:
            steps = sorted(first_steps)
        else:
            steps = list(requested_steps)
        if len(steps) == 0:
            raise FileNotFoundError(f"No saved predictions found for variable '{variable}' in {first_preds_dir}")
        expected_names = [f"{variable}_step{step}.pt" for step in steps]

    files_by_output = []
    for name in expected_names:
        paths = []
        for exp_path in exp_paths:
            path = os.path.join(exp_path, "preds", name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing prediction file: {path}")
            paths.append(path)
        files_by_output.append((name, paths))
    return files_by_output


def load_prediction(path: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if torch.is_tensor(value):
        return value.detach().cpu()
    raise TypeError(f"Unsupported prediction type in {path}: {type(value)}")


def weighted_average(paths: Sequence[str], weights: np.ndarray) -> np.ndarray:
    avg = None
    reference_shape = None
    for path, weight in zip(paths, weights):
        tensor = load_prediction(path).to(dtype=torch.float64)
        if reference_shape is None:
            reference_shape = tuple(tensor.shape)
            avg = torch.zeros_like(tensor, dtype=torch.float64)
        elif tuple(tensor.shape) != reference_shape:
            raise ValueError(
                f"Shape mismatch for {path}: got {tuple(tensor.shape)}, expected {reference_shape}"
            )
        avg.add_(tensor, alpha=float(weight))
    return avg.numpy()


def prepare_output_dir(output_dir: str, overwrite: bool):
    preds_dir = os.path.join(output_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    return preds_dir


def copy_common_metadata(exp_paths: Sequence[str], output_dir: str, init_times: Optional[List[str]]):
    if init_times is not None:
        with open(os.path.join(output_dir, "init_times.json"), "w") as f:
            json.dump(init_times, f, indent=4)

    metadata_path = os.path.join(exp_paths[0], "predict_all_metadata.json")
    if os.path.exists(metadata_path):
        shutil.copy2(metadata_path, os.path.join(output_dir, "predict_all_metadata.json"))


def write_ensemble_metadata(
    output_dir: str,
    exp_paths: Sequence[str],
    weights: np.ndarray,
    variables: Sequence[str],
    requested_steps: Optional[Sequence[int]],
    saved_files: Sequence[str],
):
    metadata = {
        "mode": "weighted_prediction_ensemble",
        "experiments": list(exp_paths),
        "weights": weights.tolist(),
        "variables": list(variables),
        "steps": list(requested_steps) if requested_steps is not None else "auto",
        "saved_files": list(saved_files),
    }
    with open(os.path.join(output_dir, "ensemble_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    args = parse_args()
    exp_paths = resolve_experiment_paths(args.root, args.experiments)
    weights = normalize_weights(args.weights, len(exp_paths))
    init_times = check_init_times(exp_paths, args.allow_mismatched_init_times)
    preds_dir = prepare_output_dir(args.output_dir, args.overwrite)

    saved_files = []
    for variable in args.variables:
        files_by_output = resolve_variable_files(exp_paths, variable, args.steps)
        for output_name, input_paths in files_by_output:
            avg = weighted_average(input_paths, weights)
            output_path = os.path.join(preds_dir, output_name)
            torch.save(avg, output_path, pickle_protocol=4)
            saved_files.append(os.path.join("preds", output_name))
            print(f"saved {output_path}")

    copy_common_metadata(exp_paths, args.output_dir, init_times)
    write_ensemble_metadata(
        args.output_dir,
        exp_paths,
        weights,
        args.variables,
        args.steps,
        saved_files,
    )

    print("\n" + "*" * 10 + " Done " + "*" * 10)


if __name__ == "__main__":
    main()
