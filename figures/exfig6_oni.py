from collections import OrderedDict
from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_PNG = "exfig6_oni.png"
OUTPUT_PDF = "exfig6_oni.pdf"
NINO_PRED_FILE = Path("exfig6_nino_pred.txt")
START_ONI_RAW = 0.4656943262894657
START_ONI_SMOOTHED = 0.5204351175690253

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
    "JMA",
    "MetFrance",
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

START_DATE = "2014-10-01"
END_DATE = "2016-10-01"


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
        0.4656943262894657,
        0.6950393300892312, 0.6949961093401897, 0.5061079452032523, 0.4747561590147478, 0.5157390174319326, 0.7104213452037861, 0.9843345631454093, 1.2430605126700427, 1.4978374879659813, 1.9919487555063102, 2.232929560564986, 2.4115404552997437, 2.7930630978211277, 2.8304388065973267, 2.645573454713784, 2.3076691929065714, 1.581479451846138, 0.9373953725098257, 0.2435384782582264, -0.20742043161760348, -0.5318609620291643, -0.5638634065694866, -0.5621162781216152, -0.6686116034489541
    ])  # 第一个数是起报月份当月的真实oni，后续的24个数是用于和模式预测比对的
    truth_values = running_mean_3(truth_values_raw)
    truth_values[0] = 0.5204351175690253

    models = OrderedDict()

    ccsm4_vals = running_mean_3(np.array([
        0.4656943262894657, 
        0.6211, 0.8906, 0.9291, 0.8588, 0.6666, 0.3905, 0.2981, 0.4012, 0.6442, 1.0452, 1.2075, 1.2135
    ]))
    ccsm4_vals[0] = 0.5204351175690253
    models["CCSM4"] = (month_range(START_DATE, "2015-10-01"), ccsm4_vals)
    
    models["SINTEX-F2"] = (
        month_range(START_DATE, "2015-09-01"),
        np.array([
            0.5204351175690253, 
            0.5755299573616792, 0.6002689749255481, 0.6797903949868491, 0.6975653903587672, 0.6739979362352303, 0.6336886680722932, 0.5626631255480407, 0.4846143478639083, 0.3521599940278494, 0.2672849327111469, 0.22847824175158138
        ]),
    )

    seas5_vals = running_mean_3(np.array([
        0.4656943262894657, 0.4450103402443233, 0.6024578303450312, 0.6734184292887448, 0.751282553546389, 0.8469111810372851, 0.8920497218772425
    ]))
    seas5_vals[0] = 0.5204351175690253
    models["SEAS5"] = (month_range(START_DATE, "2015-04-01"), seas5_vals)

    geo_vals = running_mean_3(np.array([
        0.4656943262894657, 0.5810973094578155, 0.6211691519034633, 0.5909138929991122, 0.6432822669118055, 0.7034184416171503, 0.7552357250349035, 0.7797055454081483, 0.7736782634816046, 0.7526054061659742, 0.7399571633251368, 0.756614433662405, 0.7906532261592255, 0.8200295638002424, 0.8365733788971927, 0.7572444964371633, 0.7066348133673722, 0.6400664730420345, 0.5760693772258086, 0.5119177027410227, 0.4002591735021734
    ]))
    geo_vals[0] = 0.5204351175690253
    models["3D-Geoformer"] = (month_range(START_DATE, "2016-06-01"), geo_vals)

    orca_vals = running_mean_3(np.array([
        0.4656943262894657,
        0.8432870705461598, 0.970371296701663, 0.9855188232455018, 0.9920816156233505, 0.9134953973069853, 0.7863620028211821, 0.7405974287489331, 0.6385951370313362, 0.5283301890923403, 0.39152799374932395, 0.3281009966833362, 0.38368229532483183, 0.5323861959414857, 0.4907240231917021, 0.3591687131929988, 0.2418172817304635, 0.18742938981019783, 0.10265706351217438, 0.05989792336592868, 0.026721695830237334, 0.007518542680395437, 0.0068466248732280575, -0.00679734419680621, -0.028756445035320904
    ]))
    orca_vals[0] = 0.5204351175690253
    models["ORCA-DL"] = (month_range(START_DATE, END_DATE), orca_vals)

    tda_vals = running_mean_3(np.array([
        0.4656943262894657,
        0.7501442004049658, 0.7692861665858064, 0.7302433690894252, 0.7298335749026262, 0.7614736804205139, 0.7365192989880385, 0.8275536728176636, 0.8305793690336251, 0.8843397817071882, 0.8188628977649453, 0.9703999007548249, 1.0873159994380146, 1.2779613911563663, 1.2893517004535622, 1.1989807519087547, 1.1031067447888796, 1.0321550676543447, 0.8948355338642566, 0.8540997053437607, 0.6999696181166545, 0.5268315195895552, 0.309686186592376, 0.23630709172633332, 0.3110205972889879
    ]))
    tda_vals[0] = 0.5204351175690253
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

    y_min, y_max = -1.85, 2.35
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
    ax.set_yticks(np.arange(-1.5, 2.1, 0.5))

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
        ncol=8,
        frameon=False,
        fontsize=9.2,
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
