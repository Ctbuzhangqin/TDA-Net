
from _paths import DATA_ROOT, PREDICTION_ROOT, STAT_ROOT, SST_MEAN_FILE_PATH, SST_STD_FILE_PATH
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from matplotlib.colors import LinearSegmentedColormap


EXP_NAME = "tda_ensemble"
LEADS = (6, 12, 18)

EVAL_START = 2011
EVAL_END = 2023
CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020

TRUE_DIR = Path(str(DATA_ROOT / "GODAS"))
STAT_DIR = Path(str(STAT_ROOT))
PRED_DIR = Path(str(PREDICTION_ROOT / f"{EXP_NAME}/preds"))
INIT_TIMES_FILE = Path(str(PREDICTION_ROOT / f"{EXP_NAME}/init_times.json"))

DEPTHS = np.array([10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000])
LABELED_DEPTH_TICKS = {10, 200, 400, 600, 800, 1000}

LAT_GRID = np.linspace(-63.5, 63.5, 128)
LON_GRID = np.linspace(0.5, 359.5, 360)

LAT_IDX = np.where((LAT_GRID >= -5) & (LAT_GRID <= 5))[0]
LON_IDX = np.where((LON_GRID >= 0) & (LON_GRID <= 360))[0]
PLOT_LONS = LON_GRID[LON_IDX]

MEAN_STAT = np.load(STAT_DIR / "mean" / "pottmp.npy")
STD_STAT = np.load(STAT_DIR / "std" / "pottmp.npy")


def month_iter(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def as_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def denormalize_equatorial(data_3d, month):
    data_3d = as_numpy(data_3d)
    denorm_temp = data_3d * STD_STAT[month - 1] + MEAN_STAT[month - 1]
    return denorm_temp[:, LAT_IDX, :][:, :, LON_IDX]


def mean_over_valid_lat(eq_temp):
    valid = np.isfinite(eq_temp)
    counts = valid.sum(axis=1)
    sums = np.where(valid, eq_temp, 0.0).sum(axis=1)
    mean = np.full((eq_temp.shape[0], eq_temp.shape[2]), np.nan)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean


def process_equatorial_thetao(data_3d, month):
    return mean_over_valid_lat(denormalize_equatorial(data_3d, month))


def process_paired_equatorial_thetao(true_data_3d, pred_data_3d, month):
    true_eq = denormalize_equatorial(true_data_3d, month)
    pred_eq = denormalize_equatorial(pred_data_3d, month)
    common_valid = np.isfinite(true_eq) & np.isfinite(pred_eq)
    counts = common_valid.sum(axis=1)

    true_sums = np.where(common_valid, true_eq, 0.0).sum(axis=1)
    pred_sums = np.where(common_valid, pred_eq, 0.0).sum(axis=1)

    shape = (true_eq.shape[0], true_eq.shape[2])
    true_mean = np.full(shape, np.nan)
    pred_mean = np.full(shape, np.nan)
    np.divide(true_sums, counts, out=true_mean, where=counts > 0)
    np.divide(pred_sums, counts, out=pred_mean, where=counts > 0)
    return true_mean, pred_mean


def get_true_climatology():
    print("Calculating GODAS thetao climatology...")
    clim_sum = np.zeros((12, len(DEPTHS), len(LON_IDX)))
    clim_counts = np.zeros((12, len(DEPTHS), len(LON_IDX)))

    for year, month in month_iter(CLIM_START_YEAR, CLIM_END_YEAR):
        path = TRUE_DIR / "thetao" / f"{year}_{month}.npy"
        if not path.exists():
            continue
        profile = process_equatorial_thetao(np.load(path), month)
        valid = np.isfinite(profile)
        clim_sum[month - 1] += np.where(valid, profile, 0.0)
        clim_counts[month - 1] += valid

    clim_mean = np.full_like(clim_sum, np.nan)
    np.divide(clim_sum, clim_counts, out=clim_mean, where=clim_counts > 0)
    return clim_mean


def load_init_times():
    with INIT_TIMES_FILE.open() as f:
        return json.load(f)


def get_prediction_lookup(lead):
    filepath = PRED_DIR / f"thetao_step{lead}.pt"
    pred_data = torch.load(filepath, weights_only=False, map_location="cpu")
    init_times = load_init_times()

    pred_lookup = {}
    for i, init_time in enumerate(init_times):
        year, month = map(int, init_time.split("_"))
        target_date = datetime(year, month, 1) + relativedelta(months=lead)
        pred_lookup[(target_date.year, target_date.month)] = pred_data[i]

    return pred_lookup


def get_pred_climatology(pred_lookup):
    clim_sum = np.zeros((12, len(DEPTHS), len(LON_IDX)))
    clim_counts = np.zeros((12, len(DEPTHS), len(LON_IDX)))

    for year, month in month_iter(CLIM_START_YEAR, CLIM_END_YEAR):
        data = pred_lookup.get((year, month))
        if data is None:
            continue
        profile = process_equatorial_thetao(data, month)
        valid = np.isfinite(profile)
        clim_sum[month - 1] += np.where(valid, profile, 0.0)
        clim_counts[month - 1] += valid

    clim_mean = np.full_like(clim_sum, np.nan)
    np.divide(clim_sum, clim_counts, out=clim_mean, where=clim_counts > 0)
    return clim_mean


def calc_tcc_from_arrays(true_arr, pred_arr):
    valid = np.isfinite(true_arr) & np.isfinite(pred_arr)
    counts = valid.sum(axis=0)

    true_sum = np.where(valid, true_arr, 0.0).sum(axis=0)
    pred_sum = np.where(valid, pred_arr, 0.0).sum(axis=0)

    true_mean = np.full(true_arr.shape[1:], np.nan)
    pred_mean = np.full(pred_arr.shape[1:], np.nan)
    np.divide(true_sum, counts, out=true_mean, where=counts > 0)
    np.divide(pred_sum, counts, out=pred_mean, where=counts > 0)

    true_dev = np.where(valid, true_arr - true_mean, 0.0)
    pred_dev = np.where(valid, pred_arr - pred_mean, 0.0)

    cov = (true_dev * pred_dev).sum(axis=0)
    true_var = (true_dev**2).sum(axis=0)
    pred_var = (pred_dev**2).sum(axis=0)

    denom = np.sqrt(true_var * pred_var)
    tcc = np.full(true_arr.shape[1:], np.nan)
    np.divide(cov, denom, out=tcc, where=(counts >= 2) & (denom > 0))

    return np.ma.masked_invalid(tcc)


def calc_tcc_for_lead(true_clim, lead):
    print(f"Calculating {EXP_NAME} thetao TCC, lead={lead}...")
    pred_lookup = get_prediction_lookup(lead)
    pred_clim = get_pred_climatology(pred_lookup)

    true_anoms = []
    pred_anoms = []
    for year, month in month_iter(EVAL_START, EVAL_END):
        true_path = TRUE_DIR / "thetao" / f"{year}_{month}.npy"
        pred_data = pred_lookup.get((year, month))
        if not true_path.exists() or pred_data is None:
            continue

        true_profile, pred_profile = process_paired_equatorial_thetao(np.load(true_path), pred_data, month)
        true_anoms.append(true_profile - true_clim[month - 1])
        pred_anoms.append(pred_profile - pred_clim[month - 1])

    if not true_anoms:
        raise ValueError(f"No overlapping GODAS/prediction samples for lead={lead}.")

    return calc_tcc_from_arrays(np.array(true_anoms), np.array(pred_anoms))


def lon_label(lon):
    if lon in (0, 360):
        return "0°"
    if lon == 180:
        return "180°"
    if lon < 180:
        return f"{int(lon)}°E"
    return f"{int(360 - lon)}°W"


def build_tcc_colormap():
    colors = [
        "#9bdde8",
        "#ffffff",
        "#f6c1c1",
        "#ef8585",
        "#dc3d4d",
        "#b91525",
    ]
    cmap = LinearSegmentedColormap.from_list("tcc_red", colors, N=256)
    cmap.set_bad("#cfcfcf")
    return cmap


def plot_tcc(tcc_masked, lead):
    levels = np.linspace(-0.2, 1.0, 25)
    cmap = build_tcc_colormap()

    fig, ax = plt.subplots(figsize=(3.25, 2.15), dpi=300)
    ax.set_facecolor("#cfcfcf")

    ax.contourf(
        PLOT_LONS,
        DEPTHS,
        tcc_masked,
        levels=levels,
        cmap=cmap,
        extend="both",
    )

    try:
        cs = ax.contour(
            PLOT_LONS,
            DEPTHS,
            tcc_masked,
            levels=[0.5],
            colors="black",
            linewidths=1.2,
        )
        ax.clabel(cs, fmt={0.5: "0.5"}, fontsize=8, inline=True, inline_spacing=2)
    except ValueError:
        pass

    ax.set_ylim(1000, 10)
    ax.set_xlim(0, 360)

    xticks = [0, 60, 120, 180, 240, 300, 360]
    ax.set_xticks(xticks)
    ax.set_xticklabels([lon_label(x) for x in xticks])

    ax.set_yticks(DEPTHS)
    ax.set_yticklabels([str(d) if d in LABELED_DEPTH_TICKS else "" for d in DEPTHS])
    ax.set_ylabel("Depth (m)", fontsize=10)

    ax.text(
        -0.10,
        1.07,
        "E",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    ax.set_title(f"TCC; (5°S-5°N); lead = {lead}", fontsize=10, pad=8)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.1,
        labelsize=9,
        top=False,
        right=False,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    plt.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.82)

    output_stem = f"nc_thetao_tcc_{EXP_NAME}_lead{lead}"
    for ext in ("png", "pdf"):
        path = f"{output_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


def main():
    true_clim = get_true_climatology()

    for lead in LEADS:
        tcc_masked = calc_tcc_for_lead(true_clim, lead)
        plot_tcc(tcc_masked, lead)


if __name__ == "__main__":
    main()
