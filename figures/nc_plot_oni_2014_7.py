from collections import OrderedDict
from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_PNG = "nc_oni_2014_7.png"
OUTPUT_PDF = "nc_oni_2014_7.pdf"
NINO_PRED_FILE = Path("nino_pred_2014_7.txt")
START_ONI_RAW = -0.08546460788285243
START_ONI_SMOOTHED = 0.08366791314378615

BASE_MODEL_ORDER = [
    "TDA-Net",
    "ORCA-DL",
    "3D-Geoformer",
    "SEAS5",
    "SINTEX-F2",
    "CCSM4",
]
NINO_PRED_MODEL_ORDER = [
    "CESM1",
    "NASA-GEOSS2S",
    "NCEP-CFSv2",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "GFDL-SPEAR",
]
MODEL_ORDER = BASE_MODEL_ORDER + NINO_PRED_MODEL_ORDER

MODEL_STYLES = {
    "CCSM4": dict(color="#5d63d9", linestyle=(0, (5, 3)), linewidth=2.65, alpha=0.95, marker=""),
    "SINTEX-F2": dict(color="#58a5f0", linestyle=(0, (4, 2, 1, 2)), linewidth=2.65, alpha=0.95, marker=""),
    "SEAS5": dict(color="#2ca02c", linestyle=(0, (2, 2)), linewidth=2.65, alpha=0.95, marker=""),
    "3D-Geoformer": dict(color="#7a4fa3", linestyle="-", linewidth=2.65, alpha=0.95, marker="^"),
    "TDA-Net": dict(color="#ff0000", linestyle="-", linewidth=2.65, alpha=0.95, marker="s"),
    "ORCA-DL": dict(color="#1f77b4", linestyle="-", linewidth=2.85, alpha=1.0, marker="D"),
    "CESM1": dict(color="#8c564b", linestyle=(0, (6, 2)), linewidth=2.25, alpha=0.90, marker=""),
    "NASA-GEOSS2S": dict(color="#ff7f0e", linestyle=(0, (3, 2, 1, 2)), linewidth=2.25, alpha=0.90, marker=""),
    "NCEP-CFSv2": dict(color="#17becf", linestyle=(0, (1, 2)), linewidth=2.25, alpha=0.90, marker=""),
    "CanSIPS-IC3": dict(color="#9467bd", linestyle=(0, (6, 2)), linewidth=2.25, alpha=0.90, marker=""),
    "CanSIPS-IC4": dict(color="#e377c2", linestyle=(0, (4, 2)), linewidth=2.25, alpha=0.90, marker=""),
    "GFDL-SPEAR": dict(color="#bcbd22", linestyle=(0, (2, 2)), linewidth=2.25, alpha=0.90, marker=""),
    "JMA": dict(color="#7f7f7f", linestyle=(0, (4, 2)), linewidth=2.25, alpha=0.90, marker=""),
    "MetFrance": dict(color="#d62728", linestyle=(0, (2, 1)), linewidth=2.25, alpha=0.90, marker=""),
}

START_DATE = "2014-07-01"
END_DATE = "2016-07-01"


def running_mean_3(arr):
    n = len(arr)
    if n <= 1:
        return arr.copy()

    result = np.empty(n)
    result[0] = np.mean(arr[:2])
    for i in range(1, n - 1):
        result[i] = np.mean(arr[i - 1:i + 2])
    result[-1] = np.mean(arr[-2:])
    return result


def month_range(start, end):
    return pd.date_range(start=start, end=end, freq="MS")


def row_major_legend_input(labels, ncol):
    nrow = int(np.ceil(len(labels) / ncol))
    ordered = []
    for col in range(ncol):
        for row in range(nrow):
            idx = row * ncol + col
            if idx < len(labels):
                ordered.append(labels[idx])
    return ordered


def load_nino_pred_models(path=NINO_PRED_FILE):
    ignored_headers = (
        "Raw monthly",
        "No 3-month",
        "Forecast start:",
        "Future lead",
        "Therefore",
        "Raw lead",
        "Source:",
        "Climatology:",
        "Lead ",
    )
    lead_value_re = re.compile(r":\s*([-+]?\d+(?:\.\d+)?)\s*$")
    models = OrderedDict()
    current_name = None
    current_values = []

    def flush_current():
        if current_name is not None and current_values:
            models[current_name] = np.array(current_values, dtype=float)

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue

        if text.startswith("#"):
            header = text[1:].strip()
            if header and not header.startswith(ignored_headers):
                flush_current()
                current_name = "CESM1" if header == "COLA-RSMAS-CESM1" else header
                current_values = []
            continue

        if "," not in text and not text.startswith("Lead ") and ":" not in text:
            flush_current()
            current_name = text
            current_values = []
            continue

        if current_name is None:
            continue

        match = lead_value_re.search(text)
        if match:
            current_values.append(float(match.group(1)))
        elif "," in text:
            current_values.extend(float(value) for value in text.split(",") if value.strip())

    flush_current()
    return models


def build_data():
    truth_dates = month_range(START_DATE, END_DATE)
    truth_values_raw = np.array([
        START_ONI_RAW,
        0.06849452518587888, 0.4005716963283787, 0.4656943262894657, 0.6950393300892312, 0.6949961093401897, 0.5061079452032523, 0.4747561590147478, 0.5157390174319326, 0.7104213452037861, 0.9843345631454093, 1.2430605126700427, 1.4978374879659813, 1.9919487555063102, 2.232929560564986, 2.4115404552997437, 2.7930630978211277, 2.8304388065973267, 2.645573454713784, 2.3076691929065714, 1.581479451846138, 0.9373953725098257, 0.2435384782582264, -0.20742043161760348, -0.5318609620291643
    ])  # 第一个数是起报月份当月的真实oni，后续的24个数是用于和模式预测比对的
    truth_values = running_mean_3(truth_values_raw)
    truth_values[0] = START_ONI_SMOOTHED

    models = OrderedDict()

    ccsm4_vals = running_mean_3(np.array([
        START_ONI_RAW, 
        0.2995, 0.4970, 0.9048, 1.0306, 1.2183, 1.3558, 1.1551, 0.9471, 0.6873, 0.5329, 0.3854, 0.2101
    ]))
    ccsm4_vals[0] = START_ONI_SMOOTHED
    models["CCSM4"] = (month_range(START_DATE, "2015-07-01"), ccsm4_vals)

    models["SINTEX-F2"] = (
        month_range(START_DATE, "2015-06-01"),
        np.array([
            START_ONI_SMOOTHED, 
            0.8128483906994006, 0.8445644369370449, 0.9308149360345865, 1.0016544830641116, 1.0308445894573002, 0.9834036165988861, 0.9028443644962584, 0.8381797621278638, 0.7253467732936135, 0.5510554708285333, 0.4257966854994421
        ]),
    )

    seas5_vals = running_mean_3(np.array([
        START_ONI_RAW, 0.11234585349092185, 0.2607708372420916, 0.40459477349054396, 0.5235253049792732, 0.5831560508475491, 0.6160712455822477
    ]))
    seas5_vals[0] = START_ONI_SMOOTHED
    models["SEAS5"] = (month_range(START_DATE, "2015-01-01"), seas5_vals)

    geo_vals = running_mean_3(np.array([
        START_ONI_RAW, 0.15701770772062762, 0.33070404628679817, 0.4193204027744531, 0.4595494806094333, 0.4879651043379608, 0.47808815170837243, 0.5290712877069624, 0.5769386886752066, 0.6332510094381316, 0.6792224088805567, 0.7171131753238598, 0.7371896604770941, 0.7641666095074922, 0.8348481550421435, 0.9108186933043603, 0.9769635950173204, 1.0198923595575273, 0.9799476026943768, 0.9068446549542268, 0.7991177127766664
    ]))
    geo_vals[0] = START_ONI_SMOOTHED
    models["3D-Geoformer"] = (month_range(START_DATE, "2016-03-01"), geo_vals)

    orca_vals = running_mean_3(np.array([
        START_ONI_RAW,
        -0.037463708777095865, 0.02196506865802721, 0.12246637688539638, 0.16046095160462848, 0.2986690352066188, 0.34935685474612865, 0.542026774053166, 0.6096558312319084, 0.7037480526247274, 0.7552656353122376, 0.8870328239647065, 0.9539297470967838, 1.2153012272900556, 1.2800801398359771, 1.4181094865400459, 1.5780874625569346, 1.7383286532368718, 1.722658605902132, 1.7577292924220296, 1.5489887823309778, 1.2759618938865132, 0.9869597760830435, 0.563329951388197, 0.03197401723505683
    ]))
    orca_vals[0] = START_ONI_SMOOTHED
    models["ORCA-DL"] = (month_range(START_DATE, END_DATE), orca_vals)

    tda_vals = running_mean_3(np.array([
        START_ONI_RAW,
        -0.12505074096471563, 0.021148347080802914, 0.12425709678162354, 0.2084305554570185, 0.3708262353514497, 0.39795129851345945, 0.5499243208953878, 0.6480082825496002, 0.739311557124727, 0.8630603775633042, 1.1804665787784956, 1.379979003245204, 1.7048535861941594, 1.8440144767015993, 2.21392628104696, 2.371056134836356, 2.6591967472698714, 2.5571144871713285, 2.4172296005669076, 2.0867573933743264, 1.7611298534573192, 1.44704512377685, 0.8765053776956729, -0.06812408057672951
    ]))
    tda_vals[0] = START_ONI_SMOOTHED
    models["TDA-Net"] = (month_range(START_DATE, END_DATE), tda_vals)

    for name, raw_values in load_nino_pred_models().items():
        values = running_mean_3(np.concatenate(([START_ONI_RAW], raw_values)))
        values[0] = START_ONI_SMOOTHED
        models[name] = (pd.date_range(start=START_DATE, periods=len(values), freq="MS"), values)

    return truth_dates, truth_values, models


def plot_oni():
    truth_dates, truth_values, models = build_data()

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

    fig, ax = plt.subplots(figsize=(15.2, 4.15), dpi=300)

    y_min, y_max = -0.45, 3.35
    # ax.axhspan(0.5, y_max, color="#d73027", alpha=0.055, zorder=0)
    # ax.axhspan(y_min, -0.5, color="#2c7bb6", alpha=0.055, zorder=0)
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
        kw = MODEL_STYLES[name]
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
        label="Observation",
        zorder=6,
    )

    ax.set_xlim(pd.Timestamp(START_DATE) - pd.DateOffset(days=16),
                pd.Timestamp(END_DATE) + pd.DateOffset(days=18))
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
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=1.15, color="0.70", zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    ax.text(
        0.0,
        1.065,
        "A",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        0.035,
        1.065,
        f"ONI prediction initialized in {pd.Timestamp(START_DATE):%b %Y}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
    )
    # ax.text(
    #     0.012,
    #     0.06,
    #     "La Nina threshold",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="bottom",
    #     fontsize=10.5,
    #     color="#2c7bb6",
    # )
    # ax.text(
    #     0.988,
    #     0.93,
    #     "El Nino threshold",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=10.5,
    #     color="#d73027",
    # )

    handles, labels = ax.get_legend_handles_labels()
    order = ["Observation"] + MODEL_ORDER
    order = [name for name in order if name in labels]
    row_major_order = row_major_legend_input(order, ncol=8)
    ordered = [(handles[labels.index(name)], name) for name in row_major_order]
    ax.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        ncol=7,
        frameon=False,
        fontsize=12,
        handlelength=1.7,
        columnspacing=0.65,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.055, right=0.992, top=0.93, bottom=0.215)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_oni()
    print(f"Saved {OUTPUT_PNG} and {OUTPUT_PDF}")
