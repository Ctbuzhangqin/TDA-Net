
from _paths import DATA_ROOT, PREDICTION_ROOT, STAT_ROOT, SST_MEAN_FILE_PATH, SST_STD_FILE_PATH
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
import torch
from shapely.geometry import box

import nc_plot_oni_2014_7 as oni
import nc_plot_ssta_2014_7 as base
import nc_plot_ssta_perspective as persp


OUTPUT_PNG = "nc_combined_oni_ssta_thetao_2014_7.png"
OUTPUT_PDF = "nc_combined_oni_ssta_thetao_2014_7.pdf"
DPI = 300
FIGSIZE = (17.2, 18.4)

TITLE_SIZE = 16
PANEL_LABEL_SIZE = 19
INTERNAL_SCALE = 1.0
# OBS_MONTHS = [(2015, month) for month in range(5, 11)]
OBS_MONTHS = [(2015, 10), (2015, 11), (2015, 12), (2016, 1), (2016, 2), (2016, 3)]
PRED_LEADS = range(15, 21)
TRUE_DIR = Path(str(DATA_ROOT / "GODAS"))
PANEL_LABELS = list("abcdefghijklm")
PERSPECTIVE_Y_SCALE = 0.80
PERSPECTIVE_Y_CENTER = 0.57


def scale_perspective_y(y):
    return PERSPECTIVE_Y_CENTER + PERSPECTIVE_Y_SCALE * (np.asarray(y) - PERSPECTIVE_Y_CENTER)


def warp_top(lon, lat):
    x, y = persp.warp_top(lon, lat)
    return x, scale_perspective_y(y)


def warp_section(lon, depth):
    x, y = persp.warp_section(lon, depth)
    return x, scale_perspective_y(y)


def add_panel_label(ax, label):
    offset = ScaledTranslation(-18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, label,
        transform=ax.transAxes + offset,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_panel_title(ax, title):
    title_offset = ScaledTranslation(18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, title,
        transform=ax.transAxes + title_offset,
        fontsize=TITLE_SIZE,
        ha="left",
        va="bottom",
    )


def configure_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.unicode_minus": False,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def draw_oni_panel(ax):
    truth_dates, truth_values, models = oni.build_data()

    y_min, y_max = -0.45, 3.35
    ax.axhline(0.5, color="#d73027", linestyle=(0, (5, 4)), linewidth=1.25, alpha=0.85)
    ax.axhline(-0.5, color="#2c7bb6", linestyle=(0, (5, 4)), linewidth=1.25, alpha=0.85)
    ax.axhline(0.0, color="0.15", linestyle="-", linewidth=1.05, alpha=0.75)

    ax.fill_between(
        truth_dates,
        0.5,
        truth_values,
        where=truth_values >= 0.5,
        color="#d73027",
        alpha=0.16,
        interpolate=True,
        zorder=1,
    )
    ax.fill_between(
        truth_dates,
        -0.5,
        truth_values,
        where=truth_values <= -0.5,
        color="#2c7bb6",
        alpha=0.16,
        interpolate=True,
        zorder=1,
    )

    for name, (dates, values) in models.items():
        kw = oni.MODEL_STYLES[name]
        ax.plot(
            dates,
            values,
            label=name,
            color=kw["color"],
            linestyle=kw["linestyle"],
            linewidth=kw["linewidth"],
            alpha=kw["alpha"],
            marker=kw["marker"],
            markersize=4.2 if kw["marker"] else 0,
            markevery=2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=4 if name in {"ORCA-DL", "TDA-Net"} else 3,
        )

    ax.plot(
        truth_dates,
        truth_values,
        color="black",
        linewidth=3.0,
        marker="o",
        markersize=4.6,
        markevery=2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="GODAS",
        zorder=6,
    )

    ax.set_xlim(pd.Timestamp(oni.START_DATE) - pd.DateOffset(days=16),
                pd.Timestamp(oni.END_DATE) + pd.DateOffset(days=18))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("ONI (°C)", fontsize=16, labelpad=4)
    ax.set_xlabel("Time (month)", fontsize=16, labelpad=6)
    ax.set_yticks(np.arange(-0.0, 3.1, 0.5))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="both", which="major", length=6.5, labelsize=14, pad=6)
    ax.tick_params(axis="x", which="minor", length=3.5)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    handles, labels = ax.get_legend_handles_labels()
    order = ["GODAS"] + oni.MODEL_ORDER
    order = [name for name in order if name in labels]
    row_major_order = oni.row_major_legend_input(order, ncol=8)
    ordered = [(handles[labels.index(name)], name) for name in row_major_order]
    ax.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        ncol=8,
        frameon=False,
        fontsize=13,
        handlelength=2.0,
        columnspacing=0.85,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


def compute_true_tos_climatology(lat_idx, lon_idx):
    sst_mean = np.load(base.SST_MEAN_FILE)
    sst_std = np.load(base.SST_STD_FILE)
    clim_sum = np.zeros((12, len(lat_idx), len(lon_idx)), dtype=np.float64)
    clim_count = np.zeros((12, len(lat_idx), len(lon_idx)), dtype=np.float64)
    for year in range(base.CLIM_START_YEAR, base.CLIM_END_YEAR + 1):
        for month in range(1, 13):
            path = TRUE_DIR / "tos" / f"{year}_{month}.npy"
            if not path.exists():
                continue
            field = base.denormalize_tos_field(np.load(path), month, sst_mean, sst_std)
            field = field[np.ix_(lat_idx, lon_idx)]
            valid = np.isfinite(field)
            clim_sum[month - 1] += np.where(valid, field, 0.0)
            clim_count[month - 1] += valid
    clim = np.full_like(clim_sum, np.nan)
    np.divide(clim_sum, clim_count, out=clim, where=clim_count > 0)
    return clim


def compute_true_theta_climatology(eq_lat_idx, lon_idx):
    clim_sum = np.zeros((12, len(base.DEPTHS), len(lon_idx)), dtype=np.float64)
    clim_count = np.zeros((12, len(base.DEPTHS), len(lon_idx)), dtype=np.float64)
    theta_mean = np.load(base.THETA_MEAN_FILE)
    theta_std = np.load(base.THETA_STD_FILE)
    for year in range(base.CLIM_START_YEAR, base.CLIM_END_YEAR + 1):
        for month in range(1, 13):
            path = TRUE_DIR / "thetao" / f"{year}_{month}.npy"
            if not path.exists():
                continue
            field = base.denormalize_theta_field(np.load(path), month, theta_mean, theta_std)
            profile = base.extract_equatorial_profile(field, eq_lat_idx, lon_idx)
            valid = np.isfinite(profile)
            clim_sum[month - 1] += np.where(valid, profile, 0.0)
            clim_count[month - 1] += valid
    clim = np.full_like(clim_sum, np.nan)
    np.divide(clim_sum, clim_count, out=clim, where=clim_count > 0)
    return clim


def extract_godas_anomaly(year, month, context):
    sst = base.denormalize_tos_field(
        np.load(TRUE_DIR / "tos" / f"{year}_{month}.npy"),
        month,
        context["sst_mean"],
        context["sst_std"],
    )
    ssta = sst[np.ix_(context["lat_idx"], context["lon_idx"])] - context["true_tos_clim"][month - 1]

    theta = base.denormalize_theta_field(
        np.load(TRUE_DIR / "thetao" / f"{year}_{month}.npy"),
        month,
        context["theta_mean"],
        context["theta_std"],
    )
    theta_profile = base.extract_equatorial_profile(theta, context["eq_lat_idx"], context["lon_idx"])
    theta_anom = theta_profile - context["true_theta_clim"][month - 1]

    return {
        "target_year": year,
        "target_month": month,
        "lat_idx": context["lat_idx"],
        "lon_idx": context["lon_idx"],
        "ssta": ssta,
        "theta_anom": theta_anom,
    }


def make_context():
    init_times = base.load_init_times()
    if base.TARGET_INIT_TIME not in init_times:
        raise ValueError(f"{base.TARGET_INIT_TIME} is not in {base.INIT_TIMES_FILE}.")
    target_idx = init_times.index(base.TARGET_INIT_TIME)

    min_abs_lat = np.min(np.abs(base.LAT_GRID))
    eq_lat_idx = np.where(np.isclose(np.abs(base.LAT_GRID), min_abs_lat))[0]
    lat_idx = np.where((base.LAT_GRID >= base.LAT_MIN) & (base.LAT_GRID <= base.LAT_MAX))[0]
    lon_idx = np.where((base.LON_GRID >= base.LON_MIN) & (base.LON_GRID <= base.LON_MAX))[0]

    return {
        "init_times": init_times,
        "target_idx": target_idx,
        "sst_mean": np.load(base.SST_MEAN_FILE),
        "sst_std": np.load(base.SST_STD_FILE),
        "theta_mean": np.load(base.THETA_MEAN_FILE),
        "theta_std": np.load(base.THETA_STD_FILE),
        "lat_idx": lat_idx,
        "lon_idx": lon_idx,
        "eq_lat_idx": eq_lat_idx,
        "true_tos_clim": compute_true_tos_climatology(lat_idx, lon_idx),
        "true_theta_clim": compute_true_theta_climatology(eq_lat_idx, lon_idx),
    }


def extract_anomaly_for_lead(lead, context):
    pred_times = base.prediction_times(context["init_times"], lead)
    target_year, target_month = pred_times[context["target_idx"]]

    tos_pred = torch.load(base.PRED_DIR / f"tos_step{lead}.pt", weights_only=False, map_location="cpu")
    theta_pred = torch.load(base.PRED_DIR / f"thetao_step{lead}.pt", weights_only=False, map_location="cpu")

    tos_clim = base.compute_tos_pred_climatology(
        tos_pred, pred_times, context["sst_mean"], context["sst_std"]
    )
    sst_target = base.denormalize_tos_field(
        tos_pred[context["target_idx"]], target_month, context["sst_mean"], context["sst_std"]
    )
    ssta = sst_target - tos_clim[target_month - 1]

    theta_clim = base.compute_theta_equator_pred_climatology(
        theta_pred,
        pred_times,
        context["theta_mean"],
        context["theta_std"],
        context["eq_lat_idx"],
        context["lon_idx"],
    )
    theta_target = base.denormalize_theta_field(
        theta_pred[context["target_idx"]], target_month, context["theta_mean"], context["theta_std"]
    )
    theta_profile = base.extract_equatorial_profile(theta_target, context["eq_lat_idx"], context["lon_idx"])
    theta_anom = theta_profile - theta_clim[target_month - 1]

    return {
        "target_year": target_year,
        "target_month": target_month,
        "lat_idx": context["lat_idx"],
        "lon_idx": context["lon_idx"],
        "ssta": ssta[np.ix_(context["lat_idx"], context["lon_idx"])],
        "theta_anom": theta_anom,
    }


def clipped_land_geometries():
    clip_boxes = [
        (box(base.LON_MIN, base.LAT_MIN, 180, base.LAT_MAX), 0.0),
        (box(-180, base.LAT_MIN, base.LON_MAX - 360, base.LAT_MAX), 360.0),
    ]
    for geom in base.iter_land_polygons():
        for clip_box, lon_offset in clip_boxes:
            clipped = geom.intersection(clip_box)
            if clipped.is_empty:
                continue
            if clipped.geom_type == "Polygon":
                yield clipped, lon_offset
            elif clipped.geom_type == "MultiPolygon":
                for part in clipped.geoms:
                    yield part, lon_offset


def add_land(ax):
    for geom, lon_offset in clipped_land_geometries():
        coords = np.asarray(geom.exterior.coords, dtype=float)
        lon = coords[:, 0] + lon_offset
        lat = coords[:, 1]
        x, y = warp_top(lon, lat)
        ax.add_patch(MplPolygon(
            np.column_stack([x, y]),
            closed=True,
            facecolor=base.MISSING_COLOR,
            edgecolor=persp.EDGE_COLOR,
            linewidth=0.55,
            zorder=8,
        ))


def lon_label(lon):
    if lon == 180:
        return "180°"
    if lon < 180:
        return f"{int(lon)}°E"
    return f"{int(360 - lon)}°W"


def draw_grid_and_labels(ax, scale=1.0):
    lon_ticks = [120, 150, 180, 210, 240, 270]
    lat_ticks = [-20, -10, 0, 10, 20]
    depth_ticks = base.DEPTHS[base.DEPTHS <= base.PLOT_DEPTH_MAX]

    tick_fs = 14.0 * scale
    depth_fs = 16.0 * scale

    for lon in lon_ticks:
        lat_line = np.linspace(base.LAT_MIN, base.LAT_MAX, 160)
        x, y = warp_top(np.full_like(lat_line, lon), lat_line)
        ax.plot(x, y, color="0.35", linestyle=":", linewidth=0.45, alpha=0.55, zorder=7)

        depth_line = np.linspace(10, base.PLOT_DEPTH_MAX, 160)
        xs, ys = warp_section(np.full_like(depth_line, lon), depth_line)
        ax.plot(xs, ys, color="0.35", linestyle=":", linewidth=0.45, alpha=0.55, zorder=7)

        xeq, yeq = warp_top(lon, 0.0)
        xtop, ytop = warp_section(lon, 10.0)
        connector_color = "0.25" if lon in (lon_ticks[0], lon_ticks[-1]) else "0.45"
        connector_alpha = 1.0 if lon in (lon_ticks[0], lon_ticks[-1]) else 0.70
        ax.plot(
            [xeq, xtop], [yeq, ytop],
            color=connector_color,
            linestyle=(0, (3, 3)),
            linewidth=0.65,
            alpha=connector_alpha,
            zorder=6,
        )

        xt, yt = warp_top(lon, base.LAT_MAX)
        lon_text_dx = 0.03 if lon == lon_ticks[0] else 0.0
        ax.text(
            xt + lon_text_dx, yt + 0.006, lon_label(lon),
            ha="center", va="bottom", fontsize=tick_fs, fontweight="bold"
        )

    for lat in lat_ticks:
        lon_line = np.linspace(base.LON_MIN, base.LON_MAX, 220)
        x, y = warp_top(lon_line, np.full_like(lon_line, lat))
        ax.plot(x, y, color="0.35", linestyle=":", linewidth=0.45, alpha=0.55, zorder=7)

        xl, yl = warp_top(base.LON_MIN, lat)
        label = "0°" if lat == 0 else f"{abs(lat)}°{'N' if lat > 0 else 'S'}"
        ax.text(xl - 0.008, yl, label, ha="right", va="center", fontsize=tick_fs, fontweight="bold")

    for dep in depth_ticks:
        xl, yl = warp_section(base.LON_MIN, dep)
        ax.plot([xl - 0.006, xl], [yl, yl], color="black", linewidth=0.75, zorder=11)
        if int(dep) in base.LABELED_DEPTH_TICKS:
            ax.text(xl - 0.008, yl, f"{int(dep)}", ha="right", va="center",
                    fontsize=tick_fs, fontweight="bold")

    xdep, ydep = warp_section(base.LON_MIN, 140)
    xdep0, ydep0 = warp_section(base.LON_MIN, 10)
    xdep1, ydep1 = warp_section(base.LON_MIN, base.PLOT_DEPTH_MAX)
    depth_angle = np.degrees(np.arctan2(ydep1 - ydep0, xdep1 - xdep0))
    ax.text(
        xdep - 0.08, ydep, "Depth (m)",
        rotation=depth_angle - 180,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontsize=depth_fs,
        fontweight="bold",
    )


def draw_frames(ax):
    lons = np.array([base.LON_MIN, base.LON_MAX, base.LON_MAX, base.LON_MIN, base.LON_MIN])
    lats = np.array([base.LAT_MIN, base.LAT_MIN, base.LAT_MAX, base.LAT_MAX, base.LAT_MIN])
    x, y = warp_top(lons, lats)
    ax.plot(x, y, color="black", linewidth=1.1, zorder=10)

    lon_line = np.linspace(base.LON_MIN, base.LON_MAX, 220)
    xtop, ytop = warp_section(lon_line, np.full_like(lon_line, 10.0))
    ax.plot(xtop, ytop, color="black", linewidth=1.1, zorder=10)

    for lon in [base.LON_MIN, base.LON_MAX]:
        depth_line = np.linspace(10, base.PLOT_DEPTH_MAX, 160)
        xs, ys = warp_section(np.full_like(depth_line, lon), depth_line)
        ax.plot(xs, ys, color="black", linewidth=1.1, zorder=10)

    xb, yb = warp_section(lon_line, np.full_like(lon_line, base.PLOT_DEPTH_MAX))
    ax.plot(xb, yb, color="black", linewidth=1.1, zorder=10)


def draw_perspective_panel(ax, data, scale=1.0):
    lons = base.LON_GRID[data["lon_idx"]]
    lats = base.LAT_GRID[data["lat_idx"]]
    lon2d, lat2d = np.meshgrid(lons, lats)
    x_top, y_top = warp_top(lon2d, lat2d)

    depth_idx = base.DEPTHS <= base.PLOT_DEPTH_MAX
    depths = base.DEPTHS[depth_idx]
    lon_sec, depth_sec = np.meshgrid(lons, depths)
    x_sec, y_sec = warp_section(lon_sec, depth_sec)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(base.MISSING_COLOR)
    ssta = np.ma.masked_invalid(data["ssta"])
    theta = np.ma.masked_invalid(data["theta_anom"][depth_idx, :])

    ax.set_facecolor(base.MISSING_COLOR)
    ax.contourf(x_top, y_top, ssta, levels=base.ANOM_LEVELS, cmap=cmap, extend="both", zorder=2)
    ax.contour(
        x_top, y_top, ssta,
        levels=base.CONTOUR_LEVELS,
        colors="black",
        linewidths=0.38,
        alpha=0.50,
        zorder=4,
    )
    add_land(ax)
    ax.contourf(x_sec, y_sec, theta, levels=base.ANOM_LEVELS, cmap=cmap, extend="both", zorder=1)
    ax.contour(
        x_sec, y_sec, theta,
        levels=base.CONTOUR_LEVELS,
        colors="black",
        linewidths=0.38,
        alpha=0.50,
        zorder=3,
    )
    draw_grid_and_labels(ax, scale=scale)
    draw_frames(ax)
    ax.set_xlim(0.035, 0.975)
    ax.set_ylim(0.29, 0.87)
    ax.axis("off")


def target_month_title(data):
    # Shift only the displayed month; leave the data dates unchanged.
    title_date = datetime(data["target_year"], data["target_month"], 1)
    return title_date.strftime("%b %Y")


def main():
    configure_style()
    context = make_context()

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.78, 4.0],
        hspace=0.12,
    )
    lower = outer[1].subgridspec(
        4,
        3,
        hspace=0.12,
        wspace=0.025,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    draw_oni_panel(ax_a)
    add_panel_label(ax_a, "a")
    add_panel_title(ax_a, f"ONI prediction initialized in {pd.Timestamp(oni.START_DATE):%b %Y}")

    panels = []
    for year, month in OBS_MONTHS:
        print(f"Processing GODAS {year}-{month:02d}...")
        data = extract_godas_anomaly(year, month, context)
        panels.append((data, f"{target_month_title(data)} (GODAS)"))
    for lead in PRED_LEADS:
        print(f"Processing lead {lead}...")
        data = extract_anomaly_for_lead(lead, context)
        panels.append((data, f"{target_month_title(data)} (TDA-Net)"))

    for idx, (data, title) in enumerate(panels):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(lower[row, col])
        draw_perspective_panel(ax, data, scale=INTERNAL_SCALE)
        add_panel_label(ax, PANEL_LABELS[idx + 1])
        add_panel_title(ax, title)

    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.972,
        bottom=0.03,
    )

    cax = fig.add_axes([0.2, 0.012, 0.6, 0.008])
    norm = mpl.colors.BoundaryNorm(base.ANOM_LEVELS, plt.get_cmap("RdBu_r").N)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r"))
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        ticks=[-4, -2, 0, 2, 4],
        extend="both",
    )
    cbar.ax.tick_params(labelsize=14, length=4, width=1.0, pad=4)
    cbar.outline.set_linewidth(1.0)
    cbar.ax.text(1.06, 0.5, "°C", transform=cbar.ax.transAxes, ha="left", va="center", fontsize=16, fontweight="bold")

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
