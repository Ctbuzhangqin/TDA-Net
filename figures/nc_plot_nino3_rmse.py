import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 1. 替换成你自己的数据
# =========================
# key: 模型名称
# value: (lead_time数组, correlation数组)
# 如果某个模型只预测到12个月，就只写到12；不用补到24
data = {
    "TDA-Net": (
        np.arange(1, 25),
        [0.27669674766201496, 0.3665991622542242, 0.43704058295994824, 0.4989344260916322, 0.5761279589448797, 0.6153677122751868, 0.6576236584442062, 0.6884276842415543, 0.7088618647824556, 0.7176044492572813, 0.7162792403362229, 0.7311330202162265, 0.7411308593984041, 0.7518277716410355, 0.7573992450176211, 0.7630966781387182, 0.7789270245667526, 0.7948546888700011, 0.8135195517226091, 0.8157815636089812, 0.8208821925066762, 0.8120856078733238, 0.814247119560705, 0.8024051496689301]
    ),
    "ORCA-DL": (
        np.arange(1, 25),
        [0.2724222193706134, 0.34566289381591064, 0.41591439138181013, 0.4827837334415111, 0.5464355152260154, 0.6061412330759207, 0.6424913793733198, 0.6660167834160878, 0.7042085912840855, 0.7344772549836567, 0.7667823308952308, 0.8089477634250791, 0.8483150433620928, 0.8721306589811991, 0.9062578064066731, 0.9254351199997407, 0.9451867045097054, 0.9633673394319888, 0.9829642279543063, 0.972489913225214, 0.9705274511354793, 0.9533320656465092, 0.9429166741219085, 0.9496213247745994]
    ),
    "3D-Geoformer": (
        np.arange(1, 21),
        [0.20956887421927753, 0.2177771813022825, 0.25616383979056234, 0.3114703669685157, 0.3689731431904863, 0.42143853383348057, 0.47257252905811165, 0.5195168909490218, 0.565009423552608, 0.6080404540234653, 0.6466335057751695, 0.6874168643909011, 0.7254884287366707, 0.7513124645129996, 0.7745527032979336, 0.8032759201556029, 0.832782362292493, 0.856876945051416, 0.8747299258446715, 0.8884955002585091]
    ),
    "SEAS5": (
        np.arange(1, 7),
        [0.26934841438026086, 0.3260076034477719, 0.3804418494560622, 0.4278971657708489, 0.4663340198540485, 0.5252373811074527]
    ),
    "SINTEX-F2": (
        np.arange(1, 12),
        [0.31599709278700655, 0.35976898564675885, 0.4149241058588096, 0.482033403891555, 0.547183513061124, 0.6018040417636388, 0.6449354809184157, 0.6857186950185132, 0.7314722220078341, 0.7749036256991639, 0.8048611784643271]
    ),
    "CCSM4": (
        np.arange(1, 13),
        [0.2245, 0.2847, 0.3354, 0.4059, 0.4687, 0.5209, 0.5656, 0.601, 0.6358, 0.6585, 0.6699, 0.6853]
    ),
    "CESM1": (
        np.arange(1, 13),
        [0.2396, 0.3375, 0.4211, 0.5119, 0.5852, 0.6294, 0.6534, 0.6772, 0.7175, 0.7585, 0.794, 0.8266]
    ),
    "NASA-GEOSS2S": (
        np.arange(1, 10),
        [0.3528, 0.6004, 0.8324, 0.8795, 0.8028, 0.7545, 0.744, 0.772, 0.8347]
    ),
    "NCEP-CFSv2": (
        np.arange(1, 11),
        [0.2201, 0.2967, 0.3695, 0.4339, 0.4845, 0.5356, 0.5848, 0.6199, 0.6556, 0.6888]
    ),
    "CanSIPS-IC3": (
        np.arange(1, 13),
        [0.2432, 0.2716, 0.3176, 0.3781, 0.4421, 0.5081, 0.581, 0.64, 0.6889, 0.7281, 0.7606, 0.7919]
    ),
    "CanSIPS-IC4": (
        np.arange(1, 13),
        [0.2666, 0.3003, 0.3434, 0.4052, 0.4786, 0.5489, 0.6218, 0.6887, 0.7495, 0.7964, 0.8301, 0.8589]
    ),
    "GFDL-SPEAR": (
        np.arange(1, 13),
        [0.2861, 0.3292, 0.3763, 0.4236, 0.4648, 0.523, 0.584, 0.6391, 0.6933, 0.7405, 0.7811, 0.8055]
    ),
}


# =========================
# 2. 颜色和线宽设置
# =========================
colors = {
    "TDA-Net": "#ff0000",
    "ORCA-DL": "#1f77b4",
    "3D-Geoformer": "#7a4fa3",
    "SEAS5": "#2ca02c",
    "SINTEX-F2": "#58a5f0",
    "CCSM4": "#5d63d9",
    "CESM1": "#8c564b",
    "NASA-GEOSS2S": "#ff7f0e",
    "NCEP-CFSv2": "#17becf",
    "CanSIPS-IC3": "#9467bd",
    "CanSIPS-IC4": "#e377c2",
    "GFDL-SPEAR": "#bcbd22",
}
linewidths = {
    "TDA-Net": 4.0,
    "ORCA-DL": 3.0,
    "3D-Geoformer": 3.0,
    "SEAS5": 3.0,
    "SINTEX-F2": 3.0,
    "CCSM4": 3.0,
    "CESM1": 3.0,
    "NASA-GEOSS2S": 3.0,
    "NCEP-CFSv2": 3.0,
    "CanSIPS-IC3": 3.0,
    "CanSIPS-IC4": 3.0,
    "GFDL-SPEAR": 3.0,
}


# =========================
# 3. 开始绘图
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(5.5, 5.1), dpi=300)

plot_order = [
    "TDA-Net",
    "ORCA-DL",
    "3D-Geoformer",
    "SEAS5",
    "SINTEX-F2",
    "CCSM4",
    "CESM1",
    "NASA-GEOSS2S",
    "NCEP-CFSv2",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "GFDL-SPEAR",
]

for name in plot_order:
    x, y = data[name]
    ax.plot(
        x, y,
        color=colors[name],
        linewidth=linewidths[name],
        solid_capstyle="butt",
        label=name,
        zorder=4 if name == "TDA-Net" else 3
    )


# =========================
# 4. 坐标轴、网格、刻度
# =========================
ax.set_xlim(0, 25)
ax.set_ylim(0, 1.25)

ax.set_xlabel("Lead time (months)", fontsize=15, labelpad=8)
ax.set_ylabel("RMSE", fontsize=15, labelpad=8)

ax.set_xticks([2, 6, 10, 14, 18, 22])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

# 小刻度，每个月一个
ax.set_xticks(np.arange(1, 25, 1), minor=True)

ax.tick_params(axis="both", which="major", direction="in",
               length=7, width=1.8, labelsize=13, pad=8)
ax.tick_params(axis="x", which="minor", direction="in",
               length=4, width=1.5)

# 灰色点状网格
ax.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=1.7,
    color="#9e9e9e",
    zorder=1
)

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_color("black")

# =========================
# 5. 标题和左上角 B
# =========================
ax.set_title("RMSE for Niño3 prediction",
             fontsize=16, pad=10)

ax.text(
    -0.05, 1.02, "B",
    transform=ax.transAxes,
    fontsize=19,
    fontweight="bold",
    va="bottom",
    ha="left"
)

# =========================
# 6. 图例：分左右两列手动排版，更接近原图
# =========================
left_labels = [
    "TDA-Net",
    "ORCA-DL",
    "3D-Geoformer",
    "SEAS5",
    "SINTEX-F2",
    "CCSM4",
]

right_labels = [
    "CESM1",
    "NASA-GEOSS2S",
    "NCEP-CFSv2",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "GFDL-SPEAR",
]

left_handles = [
    Line2D([0], [0], color=colors[name], lw=linewidths[name])
    for name in left_labels
]

right_handles = [
    Line2D([0], [0], color=colors[name], lw=linewidths[name])
    for name in right_labels
]

leg1 = ax.legend(
    left_handles,
    left_labels,
    loc="lower right",
    bbox_to_anchor=(0.6, 0.01),
    frameon=False,
    fontsize=11.5,
    handlelength=2.0,
    handletextpad=0.7,
    borderaxespad=0.0,
    labelspacing=0.35
)

ax.add_artist(leg1)

ax.legend(
    right_handles,
    right_labels,
    loc="lower right",
    bbox_to_anchor=(0.99, 0.01),
    frameon=False,
    fontsize=11.5,
    handlelength=2.0,
    handletextpad=0.7,
    borderaxespad=0.0,
    labelspacing=0.35
)

plt.tight_layout()

# 保存图片
plt.savefig("nc_nino3_rmse_skill.png", dpi=600, bbox_inches="tight")
plt.savefig("nc_nino3_rmse_skill.pdf", bbox_inches="tight")

# plt.show()