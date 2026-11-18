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
        [0.1313859774542464, 0.2214712163753462, 0.2719218041459817, 0.3082268213686384, 0.3789033395042158, 0.4159337899925496, 0.47637496228405335, 0.489708357889849, 0.5277488785296373, 0.5374675459675593, 0.5600531941441542, 0.5712289339817503, 0.6031464249560508, 0.6148914739187213, 0.6241324805550543, 0.6280815776462623, 0.6485495799487836, 0.6584688700914051, 0.6829353756603627, 0.6982223831793342, 0.7247590709669718, 0.7294697694894874, 0.7429397865138564, 0.7365287307927723]
    ),
    "ORCA-DL": (
        np.arange(1, 25),
        [0.11534953308912674, 0.20047989846225403, 0.2593096847330975, 0.3156202520611355, 0.3465532812455481, 0.393684226788632, 0.4663908513964209, 0.4930371887237799, 0.5149520083072524, 0.5410194186850341, 0.567299123503133, 0.6109011401785034, 0.6741649881126083, 0.7023870163862663, 0.7260560486046682, 0.7456412914042357, 0.7617486959562485, 0.769404552437761, 0.7949557695764937, 0.8015412972916024, 0.8117458294497469, 0.8224816945656511, 0.8253623858272485, 0.8335796667805536]
    ),
    "3D-Geoformer": (
        np.arange(1, 21),
        [0.35278628282353663, 0.3489076145016292, 0.3701960727196776, 0.40486072098614095, 0.4469323137883732, 0.4925059617071573, 0.5389601436778608, 0.5830780546067352, 0.6253424975852283, 0.6606975895279851, 0.688201723377185, 0.7192902654061784, 0.7465131940203019, 0.7602194790767907, 0.7701038897487504, 0.78772313154507, 0.8092252815113328, 0.8256978056622708, 0.8365609961901347, 0.8439185574084489]
    ),
    "SEAS5": (
        np.arange(1, 7),
        [0.39102793779727696, 0.3628121731853353, 0.3843288546442313, 0.4262450862727566, 0.4781000297903347, 0.5422795946450306]
    ),
    "SINTEX-F2": (
        np.arange(1, 12),
        [0.344098073406781, 0.37227062783768544, 0.41577695232465567, 0.4747209240016272, 0.5313054570050065, 0.5764558148260287, 0.6264744502452507, 0.6789434282091007, 0.7176048539784515, 0.7522901097885475, 0.7874807120259889]
    ),
    "CCSM4": (
        np.arange(1, 13),
        [0.4058, 0.4102, 0.4472, 0.49, 0.5123, 0.5368, 0.5807, 0.6306, 0.6726, 0.7089, 0.7413, 0.765]
    ),
    "CESM1": (
        np.arange(1, 13),
        [0.3963, 0.3827, 0.4195, 0.4755, 0.5317, 0.598, 0.6493, 0.6903, 0.7235, 0.7583, 0.7958, 0.8245]
    ),
    "NASA-GEOSS2S": (
        np.arange(1, 10),
        [0.2808, 0.3829, 0.4396, 0.4739, 0.5242, 0.5748, 0.6368, 0.6926, 0.7746]
    ),
    "NCEP-CFSv2": (
        np.arange(1, 11),
        [0.416, 0.4339, 0.4518, 0.4924, 0.5443, 0.6036, 0.6703, 0.7332, 0.7672, 0.7942]
    ),
    "CanSIPS-IC3": (
        np.arange(1, 13),
        [0.3334, 0.2728, 0.2882, 0.3497, 0.4273, 0.5118, 0.5914, 0.6583, 0.7209, 0.7735, 0.805, 0.8198]
    ),
    "CanSIPS-IC4": (
        np.arange(1, 13),
        [0.3445, 0.3183, 0.3514, 0.4032, 0.4701, 0.5374, 0.6061, 0.6675, 0.7161, 0.7591, 0.7905, 0.8169]
    ),
    "GFDL-SPEAR": (
        np.arange(1, 13),
        [0.3367, 0.3566, 0.3998, 0.447, 0.4886, 0.5361, 0.5826, 0.6339, 0.6834, 0.7281, 0.7667, 0.7977]
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
ax.set_title("RMSE for Niño4 prediction",
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
plt.savefig("nc_nino4_rmse_skill.png", dpi=600, bbox_inches="tight")
plt.savefig("nc_nino4_rmse_skill.pdf", bbox_inches="tight")