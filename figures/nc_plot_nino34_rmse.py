import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 1. 替换成你自己的数据
# =========================
# key: 模型名称
# value: (lead_time数组, RMSE数组)
# 如果某个模型只预测到12个月，就只写到12；不用补到24
data = {
    "TDA-Net": (
        np.arange(1, 25),
        [0.2590836059561977, 0.37223638686024263, 0.4179737054446541, 0.4337386599717703, 0.4885304946710971, 0.5046490073637707, 0.5393056493378692, 0.5514914829079561, 0.5865590719940593, 0.6085894248675674, 0.6262263027709464, 0.644612692770388, 0.6710258066007377, 0.6844031193419926, 0.6934520775975281, 0.700534703516256, 0.7277114260836595, 0.748968077648153, 0.7816097138906675, 0.7937519429520257, 0.8188277250342619, 0.8170800020297206, 0.8294173999930293, 0.8130881382994934]
    ),
    "ORCA-DL": (
        np.arange(1, 25),
        [0.23938277848411965, 0.32255400411176427, 0.3936756732724557, 0.4349665103830557, 0.4598783322767136, 0.49455445421606103, 0.5295372598828543, 0.5408158991630069, 0.5778885228701603, 0.6243893996784055, 0.6705632923544179, 0.7319860301387492, 0.7929521760000349, 0.8200952050358091, 0.850072491230663, 0.8666625842354304, 0.8833753080760557, 0.8974850015751071, 0.927835250507721, 0.9327653726288, 0.9407219325758251, 0.9390455183207286, 0.9348350950990675, 0.9440751288294679]
    ),
    "3D-Geoformer": (
        np.arange(1, 21),
        [0.10214949835367824, 0.17005974422947773, 0.24321872900566324, 0.31187557431656765, 0.3764981871295163, 0.4337978323851393, 0.48796209152983067, 0.5370984991970779, 0.5827834731001666, 0.6242661486078345, 0.658214478568819, 0.6941340599203388, 0.7260739635400317, 0.7452789188938851, 0.7620748621742705, 0.7873107980931312, 0.8156085522100345, 0.8382662031720822, 0.854197341902979, 0.8662816424404365]
    ),
    "SEAS5": (
        np.arange(1, 7),
        [0.2265350159628775, 0.2887723933366658, 0.35600388967640784, 0.4131514867306709, 0.4645749192518047, 0.5285582872400177]
    ),
    "SINTEX-F2": (
        np.arange(1, 12),
        [0.265979125116513, 0.32727741610283734, 0.38964646313194184, 0.45203652004849176, 0.5162719302984792, 0.5703802394308014, 0.6202811504966177, 0.6725985624067661, 0.7152356007249709, 0.7576079780707815, 0.794578324970168]
    ),
    "CCSM4": (
        np.arange(1, 13),
        [0.1399, 0.2409, 0.3229, 0.41, 0.4747, 0.5301, 0.5784, 0.6152, 0.6506, 0.6745, 0.6833, 0.6906]
    ),
    "CESM1": (
        np.arange(1, 13),
        [0.1392, 0.27, 0.3812, 0.5, 0.5837, 0.6569, 0.6949, 0.736, 0.7757, 0.8059, 0.8335, 0.853]
    ),
    "NASA-GEOSS2S": (
        np.arange(1, 10),
        [0.2492, 0.5797, 0.7492, 0.7292, 0.6982, 0.7007, 0.7329, 0.7829, 0.8592]
    ),
    "NCEP-CFSv2": (
        np.arange(1, 11),
        [0.172, 0.2631, 0.3421, 0.4006, 0.4564, 0.5181, 0.578, 0.6249, 0.6581, 0.6838]
    ),
    "CanSIPS-IC3": (
        np.arange(1, 13),
        [0.1092, 0.2243, 0.3036, 0.3869, 0.4686, 0.5407, 0.6119, 0.6719, 0.7229, 0.7631, 0.7879, 0.8068]
    ),
    "CanSIPS-IC4": (
        np.arange(1, 13),
        [0.1004, 0.193, 0.2777, 0.3591, 0.4397, 0.5155, 0.5924, 0.6653, 0.7238, 0.7724, 0.8089, 0.8391]
    ),
    "GFDL-SPEAR": (
        np.arange(1, 13),
        [0.1928, 0.29, 0.3427, 0.396, 0.4448, 0.5038, 0.5586, 0.6149, 0.6704, 0.7187, 0.7584, 0.7908]
    ),
    "JMA": (
        np.arange(1, 6),
        [0.296932, 0.360545, 0.425431, 0.482871, 0.548511]
    ),
    "MetFrance": (
        np.arange(1, 6),
        [0.296932, 0.360545, 0.425431, 0.482871, 0.548511]
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
    "JMA": "#7f7f7f",
    "MetFrance": "#d62728",
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
    "JMA": 3.0,
    "MetFrance": 3.0,
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
    "JMA",
    "MetFrance",
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
ax.set_title("RMSE for Niño3.4 prediction",
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
    "CESM1",
]

right_labels = [
    "NASA-GEOSS2S",
    "NCEP-CFSv2",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "GFDL-SPEAR",
    "JMA",
    "MetFrance",
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
plt.savefig("nc_nino34_rmse_skill.png", dpi=600, bbox_inches="tight")
plt.savefig("nc_nino34_rmse_skill.pdf", bbox_inches="tight")

# plt.show()