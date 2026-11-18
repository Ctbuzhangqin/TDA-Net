import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from shapely.geometry import box

import exfig6_ssta as base


OUTPUT_STEM = f"exfig6_ssta_thetao_perspective_{base.TARGET_INIT_TIME}_lead{base.LEAD}"

X0 = 0.08
X1 = 0.96
TOP_Y0 = 0.66
TOP_Y1 = 0.90
SECTION_TOP_GAP = 0.145
SECTION_DEPTH = 0.38
LAND_COLOR = base.MISSING_COLOR
EDGE_COLOR = "0.22"


def warp_top(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = (lon - base.LON_MIN) / (base.LON_MAX - base.LON_MIN)
    y = (lat - base.LAT_MIN) / (base.LAT_MAX - base.LAT_MIN)

    # A mild perspective: the northern edge is slightly narrower and shifted right.
    left = X0 + 0.120 * y
    right = X1 - 0.120 * y
    xw = left + x * (right - left)
    yw = TOP_Y0 + y * (TOP_Y1 - TOP_Y0)
    return xw, yw


def warp_section(lon, depth):
    lon = np.asarray(lon, dtype=float)
    depth = np.asarray(depth, dtype=float)
    x = (lon - base.LON_MIN) / (base.LON_MAX - base.LON_MIN)
    d = (depth - 10.0) / (base.PLOT_DEPTH_MAX - 10.0)

    x_eq, y_eq = warp_top(lon, 0.0)
    # Slightly taper the lower section so it reads as a vertical curtain.
    xw = x_eq + d * (0.012 - 0.024 * x)
    yw = y_eq - SECTION_TOP_GAP - d * SECTION_DEPTH
    return xw, yw


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
            facecolor=LAND_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.55,
            zorder=8,
        ))


def lon_label(lon):
    if lon == 180:
        return "180°"
    if lon < 180:
        return f"{int(lon)}°E"
    return f"{int(360 - lon)}°W"


def draw_grid_and_labels(ax):
    lon_ticks = [120, 150, 180, 210, 240, 270]
    lat_ticks = [-20, -10, 0, 10, 20]
    depth_ticks = base.DEPTHS[base.DEPTHS <= base.PLOT_DEPTH_MAX]

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
        connector_alpha = 1 if lon in (lon_ticks[0], lon_ticks[-1]) else 0.70
        ax.plot([xeq, xtop], [yeq, ytop], color=connector_color, linestyle=(0, (3, 3)), linewidth=0.65, alpha=connector_alpha, zorder=6)

        xt, yt = warp_top(lon, base.LAT_MAX)
        lon_text_dx = 0.020 if lon == lon_ticks[0] else 0.0
        ax.text(xt + lon_text_dx, yt + 0.006, lon_label(lon), ha="center", va="bottom", fontsize= 14.5, fontweight="bold")

    for lat in lat_ticks:
        lon_line = np.linspace(base.LON_MIN, base.LON_MAX, 220)
        x, y = warp_top(lon_line, np.full_like(lon_line, lat))
        ax.plot(x, y, color="0.35", linestyle=":", linewidth=0.45, alpha=0.55, zorder=7)

        xl, yl = warp_top(base.LON_MIN, lat)
        label = "0°" if lat == 0 else f"{abs(lat)}°{'N' if lat > 0 else 'S'}"
        ax.text(xl - 0.008, yl, label, ha="right", va="center", fontsize= 14.5, fontweight="bold")

    for dep in depth_ticks:
        lon_line = np.linspace(base.LON_MIN, base.LON_MAX, 220)
        xl, yl = warp_section(base.LON_MIN, dep)
        ax.plot([xl - 0.006, xl], [yl, yl], color="black", linewidth=0.75, zorder=11)
        if int(dep) in base.LABELED_DEPTH_TICKS:
            ax.text(xl - 0.008, yl, f"{int(dep)}", ha="right", va="center", fontsize= 14.5, fontweight="bold")

    xdep, ydep = warp_section(base.LON_MIN, 140)
    xdep0, ydep0 = warp_section(base.LON_MIN, 10)
    xdep1, ydep1 = warp_section(base.LON_MIN, base.PLOT_DEPTH_MAX)
    depth_angle = np.degrees(np.arctan2(ydep1 - ydep0, xdep1 - xdep0))
    ax.text(xdep - 0.06, ydep, "Depth (m)", rotation=depth_angle - 180, rotation_mode="anchor", ha="center", va="center", fontsize=15, fontweight="bold")


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


def plot_perspective(data):
    base.configure_style()

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

    fig, ax = plt.subplots(figsize=(8.6, 5.3), dpi=300)
    ax.set_facecolor(base.MISSING_COLOR)

    ax.contourf(
        x_top, y_top, ssta,
        levels=base.ANOM_LEVELS, cmap=cmap, extend="both", zorder=2
    )
    ax.contour(
        x_top, y_top, ssta,
        levels=base.CONTOUR_LEVELS, colors="black", linewidths=0.38, alpha=0.50, zorder=4
    )
    add_land(ax)

    ax.contourf(
        x_sec, y_sec, theta,
        levels=base.ANOM_LEVELS, cmap=cmap, extend="both", zorder=1
    )
    ax.contour(
        x_sec, y_sec, theta,
        levels=base.CONTOUR_LEVELS, colors="black", linewidths=0.38, alpha=0.50, zorder=3
    )

    draw_grid_and_labels(ax)
    draw_frames(ax)

    target = f"{data['target_year']}-{data['target_month']:02d}"

    if base.DRAW_COLORBAR:
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_clim(base.ANOM_LEVELS[0], base.ANOM_LEVELS[-1])
        cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", pad=0.015, aspect=28)
        cbar.set_label("Anomaly (°C)", fontsize=10.5, fontweight="bold")

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.21, 0.93)
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.985, top=0.97, bottom=0.055)
    fig.savefig(f"{OUTPUT_STEM}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_STEM}.png")
    print(f"Saved {OUTPUT_STEM}.pdf")


def main():
    data = base.extract_anomalies()
    plot_perspective(data)


if __name__ == "__main__":
    main()
