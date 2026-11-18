#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

INPUT_VARS = ["salt", "pottmp", "sst", "ucur", "vcur", "sshg", "uflx", "vflx"]
GODAS2CMIP6 = {
    "pottmp": "thetao",
    "salt": "so",
    "sshg": "zos",
    "sst": "tos",
    "ucur": "uo",
    "uflx": "tauu",
    "vcur": "vo",
    "vflx": "tauv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert processed GODAS NetCDF files to normalized monthly .npy files "
            "compatible with dataset/GODAS."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/processed_data"),
        help="Directory containing *-processed.nc files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/GODAS"),
        help="Root directory for output .npy files.",
    )
    parser.add_argument(
        "--stat-root",
        type=Path,
        default=Path("stat"),
        help="Directory containing stat/mean and stat/std.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Years to convert, for example --years 2021.",
    )
    parser.add_argument(
        "--vars",
        nargs="*",
        default=INPUT_VARS,
        choices=INPUT_VARS,
        help="Variables to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files if they already exist.",
    )
    return parser.parse_args()


def load_stats(stat_root: Path):
    stats = {"mean": {}, "std": {}}
    for kind in ("mean", "std"):
        for var in INPUT_VARS:
            stats[kind][var] = np.load(stat_root / kind / f"{var}.npy")
    return stats


def convert_units(var_name: str, data: np.ndarray) -> np.ndarray:
    # Stats are defined in: temperature °C, salinity g/kg, current m/s, ssh m, wind stress N/m^2.
    if var_name in {"pottmp", "sst"}:
        return data - 273.15
    if var_name == "salt":
        return data * 1000.0
    return data


def normalize_month(var_name: str, data: np.ndarray, month_idx: int, stats) -> np.ndarray:
    mean = stats["mean"][var_name][month_idx]
    std = stats["std"][var_name][month_idx]
    normed = (data - mean) / std
    return np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def infer_data_var(ds: xr.Dataset, expected_var: str) -> str:
    if expected_var in ds.data_vars:
        return expected_var

    candidates = [name for name in ds.data_vars if name not in {"date", "timePlot"}]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot infer data variable for {expected_var}. Found candidates: {candidates}"
        )
    return candidates[0]


def save_monthly_slices(var_name: str, year: int, ds: xr.Dataset, stats, output_root: Path, overwrite: bool):
    data_var = infer_data_var(ds, var_name)
    target_var = GODAS2CMIP6[var_name]
    target_dir = output_root / target_var
    target_dir.mkdir(parents=True, exist_ok=True)

    data = ds[data_var].values
    data = convert_units(var_name, data)

    times = ds.indexes["time"]
    if data.shape[0] != len(times):
        raise ValueError(
            f"Time dimension mismatch for {var_name} {year}: "
            f"data has {data.shape[0]} steps, time index has {len(times)} steps."
        )

    for month_idx, timestamp in enumerate(times):
        if int(timestamp.year) != year:
            raise ValueError(
                f"Unexpected timestamp {timestamp} in {var_name}-{year}-processed.nc"
            )

        month = int(timestamp.month)
        output_file = target_dir / f"{year}_{month}.npy"
        if output_file.exists() and not overwrite:
            print(f"Skip existing {output_file}")
            continue

        monthly_data = data[month_idx]
        normed = normalize_month(var_name, monthly_data, month - 1, stats)
        np.save(output_file, normed)
        print(f"Saved {output_file} {normed.shape}")


def convert_one_file(var_name: str, year: int, input_dir: Path, stats, output_root: Path, overwrite: bool):
    input_file = input_dir / f"{var_name}-{year}-processed.nc"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    print(f"Processing {input_file}")
    with xr.open_dataset(input_file) as ds:
        save_monthly_slices(var_name, year, ds, stats, output_root, overwrite)


def main():
    args = parse_args()
    stats = load_stats(args.stat_root)

    for year in args.years:
        for var_name in args.vars:
            convert_one_file(
                var_name=var_name,
                year=year,
                input_dir=args.input_dir,
                stats=stats,
                output_root=args.output_root,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
