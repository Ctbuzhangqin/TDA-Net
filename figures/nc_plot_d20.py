
from _paths import DATA_ROOT, PREDICTION_ROOT, STAT_ROOT, SST_MEAN_FILE_PATH, SST_STD_FILE_PATH
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from matplotlib.colors import TwoSlopeNorm


EXP_NAME = "TDA-Net"
LEADS = (6, 12, 18)

PLOT_START_YEAR = 2011
PLOT_END_YEAR = 2023
CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020

TRUE_DIR = Path(str(DATA_ROOT / "GODAS"))
STAT_DIR = Path(str(STAT_ROOT))
PRED_DIR = Path(str(PREDICTION_ROOT / f"{EXP_NAME}/preds"))
INIT_TIMES_FILE = Path(str(PREDICTION_ROOT / f"{EXP_NAME}/init_times.json"))

DEPTHS = np.array([10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000])
LAT_GRID = np.linspace(-63.5, 63.5, 128)
LON_GRID = np.linspace(0.5, 359.5, 360)

LAT_IDX = np.where((LAT_GRID >= -5) & (LAT_GRID <= 5))[0]
LON_IDX = np.where((LON_GRID >= 120) & (LON_GRID <= 270))[0]
PLOT_LONS = LON_GRID[LON_IDX]

MEAN_STAT = np.load(STAT_DIR / "mean" / "pottmp.npy")
STD_STAT = np.load(STAT_DIR / "std" / "pottmp.npy")


def month_iter(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def month_axis(start_year, end_year):
    return np.array([year + (month - 1) / 12 for year, month in month_iter(start_year, end_year)])


def as_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def calc_d20_vectorized(temp_3d):
    d20 = np.full(temp_3d.shape[1:], np.nan)

    for k in range(len(DEPTHS) - 1):
        t1 = temp_3d[k]
        t2 = temp_3d[k + 1]
        mask = (t1 >= 20) & (t2 < 20)
        denom = np.where(t1 == t2, 1e-5, t1 - t2)
        d20_k = DEPTHS[k] + (DEPTHS[k + 1] - DEPTHS[k]) * (t1 - 20) / denom
        d20[mask] = d20_k[mask]

    return d20


def load_and_calc_d20(data_3d, month):
    data_3d = as_numpy(data_3d)
    denorm_temp = data_3d * STD_STAT[month - 1] + MEAN_STAT[month - 1]
    eq_temp = denorm_temp[:, LAT_IDX, :]
    d20 = calc_d20_vectorized(eq_temp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d20_equator = np.nanmean(d20, axis=0)

    return d20_equator[LON_IDX]


def get_true_d20_anomaly():
    print("Calculating GODAS D20 anomaly...")
    clim_d20 = np.zeros((12, len(LON_IDX)))
    clim_counts = np.zeros(12)

    for year, month in month_iter(CLIM_START_YEAR, CLIM_END_YEAR):
        path = TRUE_DIR / "thetao" / f"{year}_{month}.npy"
        if not path.exists():
            continue
        d20 = load_and_calc_d20(np.load(path), month)
        clim_d20[month - 1] += d20
        clim_counts[month - 1] += 1

    if np.any(clim_counts == 0):
        missing = np.where(clim_counts == 0)[0] + 1
        raise RuntimeError(f"Missing GODAS climatology months: {missing.tolist()}")

    clim_d20 /= clim_counts[:, None]

    d20_anoms = []
    for year, month in month_iter(PLOT_START_YEAR, PLOT_END_YEAR):
        path = TRUE_DIR / "thetao" / f"{year}_{month}.npy"
        if path.exists():
            d20 = load_and_calc_d20(np.load(path), month)
            d20_anoms.append(d20 - clim_d20[month - 1])
        else:
            d20_anoms.append(np.full(len(LON_IDX), np.nan))

    return np.array(d20_anoms)


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


def get_pred_d20_anomaly(lead):
    print(f"Calculating {EXP_NAME} D20 anomaly, lead={lead}...")
    pred_lookup = get_prediction_lookup(lead)

    clim_d20 = np.zeros((12, len(LON_IDX)))
    clim_counts = np.zeros(12)

    for year, month in month_iter(CLIM_START_YEAR, CLIM_END_YEAR):
        data = pred_lookup.get((year, month))
        if data is None:
            continue
        d20 = load_and_calc_d20(data, month)
        clim_d20[month - 1] += d20
        clim_counts[month - 1] += 1

    for month_idx in range(12):
        if clim_counts[month_idx] > 0:
            clim_d20[month_idx] /= clim_counts[month_idx]
        else:
            clim_d20[month_idx] = np.nan

    d20_anoms = []
    for year, month in month_iter(PLOT_START_YEAR, PLOT_END_YEAR):
        data = pred_lookup.get((year, month))
        if data is None or np.isnan(clim_d20[month - 1]).all():
            d20_anoms.append(np.full(len(LON_IDX), np.nan))
        else:
            d20 = load_and_calc_d20(data, month)
            d20_anoms.append(d20 - clim_d20[month - 1])

    return np.array(d20_anoms)


def lon_label(lon):
    if lon == 180:
        return "180°"
    if lon < 180:
        return f"{int(lon)}°E"
    return f"{int(360 - lon)}°W"


def plot_d20_hovmoller(data, title, output_stem):
    time_values = month_axis(PLOT_START_YEAR, PLOT_END_YEAR)
    z = data.T
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)

    fig, ax = plt.subplots(figsize=(12, 2), dpi=300)
    ax.imshow(
        z,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="bilinear",
        extent=[time_values.min(), time_values.max(), PLOT_LONS.max(), PLOT_LONS.min()],
    )

    ax.set_xlim(PLOT_START_YEAR, PLOT_END_YEAR + 11 / 12)
    ax.set_ylim(PLOT_LONS.max(), PLOT_LONS.min())

    xticks = np.arange(PLOT_START_YEAR, PLOT_END_YEAR + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=18)

    yticks = np.arange(120, 271, 30)
    ax.set_yticks(yticks)
    ax.set_yticklabels([lon_label(y) if i % 2 == 0 else "" for i, y in enumerate(yticks)], fontsize=18)

    ax.tick_params(axis="both", which="major", direction="out", length=11, width=2.0, pad=7)

    for spine in ax.spines.values():
        spine.set_linewidth(2.2)

    ax.set_title(title, loc="left", fontsize=22, pad=14)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.82, bottom=0.28)

    for ext in ("png", "pdf"):
        path = f"{output_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


def main():
    print(f"Plot range: {PLOT_START_YEAR}-{PLOT_END_YEAR}")

    true_anom = get_true_d20_anomaly()
    plot_d20_hovmoller(
        true_anom,
        "Equatorial mean D20 anomalies from GODAS",
        f"nc_d20_godas_{PLOT_START_YEAR}_{PLOT_END_YEAR}",
    )

    for lead in LEADS:
        pred_anom = get_pred_d20_anomaly(lead)
        plot_d20_hovmoller(
            pred_anom,
            f"Equatorial mean D20 anomalies from {EXP_NAME} (lead={lead})",
            f"nc_d20_{EXP_NAME}_lead{lead}_{PLOT_START_YEAR}_{PLOT_END_YEAR}",
        )


if __name__ == "__main__":
    main()
