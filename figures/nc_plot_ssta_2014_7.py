
from _paths import DATA_ROOT, PREDICTION_ROOT, STAT_ROOT, SST_MEAN_FILE_PATH, SST_STD_FILE_PATH
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from cartopy.io import shapereader
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from shapely.geometry import box


EXP_NAME = "tda_ensemble"
TARGET_INIT_TIME = "2014_7"
LEAD = 1

CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020

PRED_DIR = Path(str(PREDICTION_ROOT / f"{EXP_NAME}/preds"))
INIT_TIMES_FILE = Path(str(PREDICTION_ROOT / f"{EXP_NAME}/init_times.json"))

SST_MEAN_FILE = Path(str(SST_MEAN_FILE_PATH))
SST_STD_FILE = Path(str(SST_STD_FILE_PATH))
THETA_MEAN_FILE = Path(str(STAT_ROOT / "mean/pottmp.npy"))
THETA_STD_FILE = Path(str(STAT_ROOT / "std/pottmp.npy"))

LAT_GRID = np.linspace(-63.5, 63.5, 128)
LON_GRID = np.linspace(0.5, 359.5, 360)
DEPTHS = np.array([10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000])
PLOT_DEPTH_MAX = 250
LABELED_DEPTH_TICKS = {10, 50, 100, 150, 200, 250}

LAT_MIN = -20
LAT_MAX = 20
LON_MIN = 120
LON_MAX = 270

ANOM_LEVELS = np.linspace(-4.0, 4.0, 25)
CONTOUR_LEVELS = [-3, -2, -1, 0, 1, 2, 3]
DRAW_COLORBAR = False
MISSING_COLOR = "#bfbfbf"
LAND_EDGE_COLOR = "0.25"


def as_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def load_init_times():
    with INIT_TIMES_FILE.open() as f:
        return json.load(f)


def target_date_from_init(init_time, lead):
    year, month = map(int, init_time.split("_"))
    target = datetime(year, month, 1) + relativedelta(months=lead)
    return target.year, target.month


def prediction_times(init_times, lead):
    return [target_date_from_init(init_time, lead) for init_time in init_times]


def month_stat(stat, month):
    return np.squeeze(stat[month - 1])


def denormalize_tos_field(field, month, mean, std):
    return as_numpy(field) * month_stat(std, month) + month_stat(mean, month)


def denormalize_theta_field(field, month, mean, std):
    return as_numpy(field) * month_stat(std, month) + month_stat(mean, month)


def compute_tos_pred_climatology(pred_data, pred_times, mean, std):
    clim_sum = np.zeros((12, len(LAT_GRID), len(LON_GRID)), dtype=np.float64)
    clim_count = np.zeros(12, dtype=np.float64)

    for i, (year, month) in enumerate(pred_times):
        if CLIM_START_YEAR <= year <= CLIM_END_YEAR:
            field = denormalize_tos_field(pred_data[i], month, mean, std)
            valid = np.isfinite(field)
            clim_sum[month - 1] += np.where(valid, field, 0.0)
            clim_count[month - 1] += 1.0

    clim = np.full_like(clim_sum, np.nan)
    for month_idx in range(12):
        if clim_count[month_idx] > 0:
            clim[month_idx] = clim_sum[month_idx] / clim_count[month_idx]
    return clim


def extract_equatorial_profile(field, eq_lat_idx, lon_idx):
    eq_field = field[:, eq_lat_idx, :][:, :, lon_idx]
    valid = np.isfinite(eq_field)
    counts = valid.sum(axis=1)
    sums = np.where(valid, eq_field, 0.0).sum(axis=1)
    profile = np.full((field.shape[0], len(lon_idx)), np.nan)
    np.divide(sums, counts, out=profile, where=counts > 0)
    return profile


def compute_theta_equator_pred_climatology(pred_data, pred_times, mean, std, eq_lat_idx, lon_idx):
    clim_sum = np.zeros((12, len(DEPTHS), len(lon_idx)), dtype=np.float64)
    clim_count = np.zeros((12, len(DEPTHS), len(lon_idx)), dtype=np.float64)

    for i, (year, month) in enumerate(pred_times):
        if CLIM_START_YEAR <= year <= CLIM_END_YEAR:
            field = denormalize_theta_field(pred_data[i], month, mean, std)
            profile = extract_equatorial_profile(field, eq_lat_idx, lon_idx)
            valid = np.isfinite(profile)
            clim_sum[month - 1] += np.where(valid, profile, 0.0)
            clim_count[month - 1] += valid

    clim = np.full_like(clim_sum, np.nan)
    np.divide(clim_sum, clim_count, out=clim, where=clim_count > 0)
    return clim


def extract_anomalies():
    init_times = load_init_times()
    if TARGET_INIT_TIME not in init_times:
        raise ValueError(f"{TARGET_INIT_TIME} is not in {INIT_TIMES_FILE}.")

    target_idx = init_times.index(TARGET_INIT_TIME)
    pred_times = prediction_times(init_times, LEAD)
    target_year, target_month = pred_times[target_idx]

    tos_pred = torch.load(PRED_DIR / f"tos_step{LEAD}.pt", weights_only=False, map_location="cpu")
    theta_pred = torch.load(PRED_DIR / f"thetao_step{LEAD}.pt", weights_only=False, map_location="cpu")

    sst_mean = np.load(SST_MEAN_FILE)
    sst_std = np.load(SST_STD_FILE)
    theta_mean = np.load(THETA_MEAN_FILE)
    theta_std = np.load(THETA_STD_FILE)

    tos_clim = compute_tos_pred_climatology(tos_pred, pred_times, sst_mean, sst_std)
    sst_target = denormalize_tos_field(tos_pred[target_idx], target_month, sst_mean, sst_std)
    ssta = sst_target - tos_clim[target_month - 1]

    lat_idx = np.where((LAT_GRID >= LAT_MIN) & (LAT_GRID <= LAT_MAX))[0]
    lon_idx = np.where((LON_GRID >= LON_MIN) & (LON_GRID <= LON_MAX))[0]
    min_abs_lat = np.min(np.abs(LAT_GRID))
    eq_lat_idx = np.where(np.isclose(np.abs(LAT_GRID), min_abs_lat))[0]

    theta_clim = compute_theta_equator_pred_climatology(
        theta_pred, pred_times, theta_mean, theta_std, eq_lat_idx, lon_idx
    )
    theta_target = denormalize_theta_field(theta_pred[target_idx], target_month, theta_mean, theta_std)
    theta_profile = extract_equatorial_profile(theta_target, eq_lat_idx, lon_idx)
    theta_anom = theta_profile - theta_clim[target_month - 1]

    return {
        "target_year": target_year,
        "target_month": target_month,
        "lat_idx": lat_idx,
        "lon_idx": lon_idx,
        "equator_lats": LAT_GRID[eq_lat_idx],
        "ssta": ssta[np.ix_(lat_idx, lon_idx)],
        "theta_anom": theta_anom,
    }


def lon_label(lon):
    if lon == 180:
        return "180°"
    if lon < 180:
        return f"{int(lon)}°E"
    return f"{int(360 - lon)}°W"





def iter_land_polygons():
    land_path = shapereader.natural_earth(
        resolution="110m",
        category="physical",
        name="land",
    )
    for geom in shapereader.Reader(land_path).geometries():
        if geom.geom_type == "Polygon":
            yield geom
        elif geom.geom_type == "MultiPolygon":
            yield from geom.geoms



def add_land_geometry(ax, geom, lon_offset=0.0):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        coords = np.asarray(geom.exterior.coords, dtype=float)
        coords[:, 0] += lon_offset
        ax.add_patch(MplPolygon(
            coords,
            closed=True,
            facecolor=MISSING_COLOR,
            edgecolor=LAND_EDGE_COLOR,
            linewidth=0.55,
            zorder=7,
            clip_on=True,
        ))
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            add_land_geometry(ax, part, lon_offset=lon_offset)


def add_natural_earth_land(ax):
    clip_boxes = [
        (box(LON_MIN, LAT_MIN, 180, LAT_MAX), 0.0),
        (box(-180, LAT_MIN, LON_MAX - 360, LAT_MAX), 360.0),
    ]
    for geom in iter_land_polygons():
        for clip_box, lon_offset in clip_boxes:
            add_land_geometry(ax, geom.intersection(clip_box), lon_offset=lon_offset)


def configure_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.unicode_minus": False,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_panels(data):
    configure_style()

    lons = LON_GRID[data["lon_idx"]]
    lats = LAT_GRID[data["lat_idx"]]
    target_label = f"{data['target_year']}-{data['target_month']:02d}"
    init_label = datetime.strptime(TARGET_INIT_TIME, "%Y_%m").strftime("%Y-%m")
    output_stem = f"nc_ssta_thetao_{TARGET_INIT_TIME}_lead{LEAD}"

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(MISSING_COLOR)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.20},
    )

    ax = axes[0]
    ax.set_facecolor(MISSING_COLOR)
    ssta = np.ma.masked_invalid(data["ssta"])
    cf1 = ax.contourf(lons, lats, ssta, levels=ANOM_LEVELS, cmap=cmap, extend="both")
    ax.contour(lons, lats, ssta, levels=CONTOUR_LEVELS, colors="black", linewidths=0.45, alpha=0.55)
    add_natural_earth_land(ax)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.45, color="0.6", alpha=0.6)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(["20°S", "10°S", "0°", "10°N", "20°N"])
    ax.set_title(
        f"Predicted SSTA; init = {init_label}; lead = {LEAD}; target = {target_label}",
        fontsize=12,
        pad=8,
    )
    ax.text(-0.08, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    ax = axes[1]
    ax.set_facecolor(MISSING_COLOR)
    depth_idx = DEPTHS <= PLOT_DEPTH_MAX
    plot_depths = DEPTHS[depth_idx]
    theta = np.ma.masked_invalid(data["theta_anom"][depth_idx, :])
    cf2 = ax.contourf(lons, plot_depths, theta, levels=ANOM_LEVELS, cmap=cmap, extend="both")
    ax.contour(lons, plot_depths, theta, levels=CONTOUR_LEVELS, colors="black", linewidths=0.45, alpha=0.55)
    ax.set_ylim(PLOT_DEPTH_MAX, 10)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_yticks(plot_depths)
    ax.set_yticklabels([str(d) if d in LABELED_DEPTH_TICKS else "" for d in plot_depths])
    ax.set_title(
        f"Predicted equatorial thetao anomaly; latitude = 0° mean; lead = {LEAD}",
        fontsize=12,
        pad=8,
    )
    ax.text(-0.08, 1.04, "B", transform=ax.transAxes, fontsize=14, fontweight="bold")
    if DRAW_COLORBAR:
        cbar = fig.colorbar(cf2, ax=axes, orientation="vertical", pad=0.015, aspect=28)
        cbar.set_label("Anomaly (°C)", fontsize=11)

    xticks = [120, 150, 180, 210, 240, 270]
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([lon_label(x) for x in xticks])

    for ax in axes:
        ax.tick_params(axis="both", which="major", length=5, width=1.2, labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")

    fig.subplots_adjust(left=0.095, right=0.91, top=0.93, bottom=0.10)
    fig.savefig(f"{output_stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_stem}.png")
    print(f"Saved {output_stem}.pdf")


def main():
    data = extract_anomalies()
    plot_panels(data)


if __name__ == "__main__":
    main()
