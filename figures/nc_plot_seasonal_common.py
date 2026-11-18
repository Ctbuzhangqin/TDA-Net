import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["hatch.linewidth"] = 1.0

INIT_MONTHS = ["May", "Jun", "Jul", "Aug", "Sep", "Oct",
               "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
MONTH_CYCLE = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TARGET_START_MONTH = "Jun"
MAX_LEAD = 24

RESULT_COLUMN_ORDER = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
                       "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
SEASON_TO_MONTH = {
    "DJF": "Jan", "JFM": "Feb", "FMA": "Mar", "MAM": "Apr",
    "AMJ": "May", "MJJ": "Jun", "JJA": "Jul", "JAS": "Aug",
    "ASO": "Sep", "SON": "Oct", "OND": "Nov", "NDJ": "Dec",
}

MODEL_TABLES = {
    "NUIST": """
Lead  DJF      JFM      FMA      MAM      AMJ      MJJ      JJA      JAS      ASO      SON      OND      NDJ
1     0.991    0.989    0.983    0.980    0.976    0.978    0.992    0.996    0.996    0.994    0.986    0.991
2     0.971    0.968    0.952    0.943    0.947    0.957    0.983    0.988    0.987    0.977    0.961    0.970
3     0.972    0.920    0.876    0.874    0.909    0.948    0.964    0.975    0.966    0.955    0.940    0.959
4     0.914    0.805    0.789    0.807    0.894    0.925    0.939    0.956    0.939    0.941    0.939    0.978
5     0.782    0.699    0.695    0.785    0.874    0.899    0.919    0.938    0.940    0.949    0.952    0.888
6     0.689    0.610    0.631    0.750    0.840    0.874    0.908    0.931    0.939    0.940    0.823    0.689
7     0.629    0.580    0.604    0.718    0.803    0.885    0.898    0.925    0.907    0.758    0.549    0.606
8     0.614    0.553    0.564    0.675    0.830    0.868    0.891    0.906    0.723    0.454    0.406    0.600
9     0.578    0.516    0.523    0.735    0.804    0.848    0.872    0.752    0.431    0.346    0.371    0.608
10    0.531    0.499    0.627    0.713    0.784    0.834    0.738    0.522    0.343    0.344    0.415    0.589
11    0.522    0.629    0.619    0.700    0.766    0.695    0.497    0.453    0.373    0.410    0.460    0.559
12    0.626    0.642    0.637    0.701    0.620    0.418    0.392    0.432    0.433    0.439    0.465    0.535
13    0.605    0.620    0.614    0.601    0.382    0.305    0.344    0.452    0.471    0.449    0.441    0.557
14    0.602    0.613    0.533    0.402    0.275    0.311    0.388    0.490    0.504    0.444    0.395    0.525
15    0.601    0.506    0.281    0.299    0.272    0.406    0.443    0.549    0.535    0.385    0.378    0.482
16    0.480    0.252    0.173    0.317    0.399    0.517    0.505    0.582    0.403    0.432    0.388    0.511
17    0.284    0.147    0.143    0.362    0.486    0.584    0.574    0.464    0.462    0.417    0.386    0.469
18    0.199    0.177    0.252    0.489    0.570    0.634    0.401    0.530    0.500    0.392    0.293    0.276
19    0.225    0.259    0.389    0.585    0.606    0.379    0.437    0.569    0.508    0.373    0.160    0.203
20    0.357    0.352    0.460    0.620    0.305    0.374    0.467    0.542    0.503    0.256    0.212    0.313
21    0.468    0.428    0.482    0.359    0.274    0.329    0.366    0.400    0.295    0.213    0.341    0.438
22    0.504    0.467    0.134    0.346    0.293    0.324    0.315    0.257    0.259    0.265    0.403    0.504
23    0.513    0.077    0.099    0.335    0.389    0.254    0.262    0.268    0.277    0.327    0.416    0.546
24    0.097    -0.012   0.041    0.367    0.466    0.095    0.227    0.285    0.353    0.475    0.531    0.558
""",
    "3D-Geoformer": """
Lead  DJF      JFM      FMA      MAM      AMJ      MJJ      JJA      JAS      ASO      SON      OND      NDJ
1     0.993    0.992    0.994    0.990    0.985    0.987    0.994    0.997    0.997    0.997    0.996    0.994    
2     0.985    0.966    0.964    0.956    0.968    0.977    0.985    0.987    0.991    0.994    0.988    0.989    
3     0.943    0.888    0.912    0.934    0.957    0.970    0.962    0.972    0.984    0.984    0.980    0.979    
4     0.835    0.809    0.888    0.918    0.948    0.948    0.940    0.962    0.972    0.974    0.966    0.927    
5     0.752    0.809    0.865    0.913    0.927    0.927    0.928    0.946    0.953    0.954    0.906    0.794    
6     0.782    0.807    0.862    0.897    0.906    0.913    0.908    0.921    0.921    0.888    0.761    0.697    
7     0.782    0.815    0.849    0.877    0.891    0.893    0.877    0.879    0.842    0.747    0.640    0.742    
8     0.781    0.809    0.832    0.860    0.861    0.859    0.824    0.788    0.708    0.641    0.677    0.760    
9     0.773    0.795    0.811    0.823    0.821    0.805    0.721    0.662    0.631    0.689    0.693    0.759    
10    0.757    0.771    0.766    0.775    0.761    0.702    0.595    0.611    0.667    0.685    0.696    0.755    
11    0.729    0.712    0.717    0.709    0.653    0.566    0.558    0.630    0.629    0.684    0.697    0.740    
12    0.623    0.661    0.652    0.593    0.498    0.514    0.598    0.584    0.636    0.693    0.692    0.713    
13    0.572    0.606    0.542    0.428    0.416    0.569    0.564    0.586    0.647    0.694    0.672    0.563    
14    0.521    0.518    0.375    0.334    0.498    0.554    0.568    0.596    0.653    0.673    0.538    0.514    
15    0.457    0.383    0.284    0.417    0.518    0.542    0.576    0.602    0.636    0.539    0.484    0.450    
16    0.347    0.313    0.375    0.454    0.511    0.545    0.586    0.587    0.500    0.483    0.400    0.372    
17    0.298    0.411    0.457    0.464    0.511    0.549    0.574    0.452    0.460    0.394    0.299    0.275    
18    0.404    0.513    0.492    0.463    0.502    0.530    0.457    0.417    0.377    0.267    0.194    0.257    
19    0.520    0.571    0.501    0.452    0.474    0.414    0.417    0.335    0.231    0.135    0.191    0.389    
20    0.589    0.591    0.490    0.421    0.303    0.370    0.332    0.196    0.085    0.126    0.331    0.518    
""",
    "ORCA-DL": """
Lead  DJF      JFM      FMA      MAM      AMJ      MJJ      JJA      JAS      ASO      SON      OND      NDJ
1     0.984    0.985    0.983    0.954    0.964    0.976    0.985    0.987    0.989    0.990    0.986    0.986    
2     0.968    0.964    0.984    0.957    0.951    0.974    0.991    0.988    0.986    0.983    0.972    0.976    
3     0.928    0.932    0.960    0.926    0.940    0.981    0.978    0.981    0.978    0.965    0.957    0.957    
4     0.823    0.851    0.910    0.912    0.952    0.971    0.966    0.974    0.959    0.957    0.949    0.915    
5     0.701    0.808    0.866    0.918    0.948    0.963    0.958    0.961    0.954    0.950    0.899    0.772    
6     0.668    0.764    0.861    0.916    0.944    0.959    0.943    0.951    0.936    0.897    0.737    0.622    
7     0.690    0.790    0.867    0.919    0.944    0.949    0.927    0.923    0.870    0.753    0.616    0.666    
8     0.739    0.816    0.867    0.915    0.931    0.932    0.894    0.848    0.723    0.635    0.617    0.677    
9     0.774    0.822    0.856    0.903    0.903    0.890    0.811    0.694    0.616    0.576    0.573    0.712    
10    0.790    0.816    0.869    0.873    0.844    0.783    0.638    0.564    0.527    0.467    0.579    0.743    
11    0.779    0.843    0.836    0.811    0.723    0.589    0.514    0.514    0.452    0.471    0.608    0.751    
12    0.774    0.815    0.773    0.678    0.504    0.429    0.461    0.415    0.418    0.467    0.605    0.734    
13    0.738    0.755    0.655    0.495    0.378    0.433    0.433    0.426    0.454    0.502    0.604    0.667    
14    0.684    0.655    0.475    0.370    0.384    0.417    0.460    0.465    0.484    0.499    0.512    0.631    
15    0.581    0.482    0.352    0.369    0.368    0.428    0.484    0.489    0.492    0.382    0.443    0.546    
16    0.427    0.346    0.378    0.337    0.380    0.450    0.501    0.500    0.403    0.323    0.351    0.449    
17    0.327    0.387    0.372    0.346    0.411    0.470    0.510    0.438    0.371    0.238    0.274    0.322    
18    0.403    0.412    0.365    0.356    0.421    0.469    0.442    0.401    0.284    0.173    0.241    0.294    
19    0.492    0.462    0.421    0.403    0.442    0.367    0.405    0.314    0.192    0.183    0.311    0.427    
20    0.551    0.520    0.462    0.411    0.262    0.334    0.321    0.194    0.159    0.272    0.416    0.506    
21    0.582    0.534    0.458    0.122    0.234    0.268    0.209    0.139    0.239    0.386    0.478    0.561    
22    0.584    0.524    0.103    0.085    0.188    0.197    0.155    0.211    0.357    0.439    0.520    0.582    
23    0.589    0.189    0.080    0.062    0.162    0.143    0.220    0.355    0.431    0.502    0.556    0.593    
24    0.293    0.136    0.024    0.043    0.103    0.172    0.349    0.427    0.484    0.537    0.564    0.582  
""",
    "TDA-Net": """
Lead  DJF      JFM      FMA      MAM      AMJ      MJJ      JJA      JAS      ASO      SON      OND      NDJ
1     0.987    0.983    0.987    0.969    0.966    0.975    0.983    0.988    0.988    0.990    0.988    0.989    
2     0.971    0.959    0.969    0.940    0.931    0.957    0.979    0.983    0.980    0.981    0.975    0.980    
3     0.926    0.908    0.939    0.913    0.918    0.958    0.962    0.975    0.973    0.968    0.966    0.966    
4     0.829    0.841    0.894    0.899    0.928    0.950    0.954    0.972    0.963    0.965    0.960    0.926    
5     0.760    0.822    0.853    0.901    0.919    0.944    0.945    0.958    0.950    0.954    0.916    0.810    
6     0.756    0.792    0.852    0.888    0.904    0.936    0.929    0.946    0.931    0.910    0.787    0.719    
7     0.755    0.812    0.845    0.876    0.899    0.925    0.904    0.918    0.881    0.784    0.685    0.754    
8     0.780    0.827    0.841    0.878    0.897    0.903    0.869    0.859    0.749    0.650    0.663    0.747    
9     0.804    0.830    0.835    0.857    0.851    0.847    0.799    0.729    0.640    0.637    0.663    0.776    
10    0.815    0.829    0.811    0.814    0.780    0.761    0.673    0.628    0.611    0.611    0.694    0.803    
11    0.824    0.822    0.783    0.757    0.683    0.610    0.580    0.631    0.610    0.664    0.743    0.824    
12    0.792    0.787    0.728    0.659    0.528    0.501    0.595    0.626    0.645    0.699    0.757    0.824    
13    0.745    0.733    0.631    0.513    0.414    0.543    0.638    0.671    0.685    0.717    0.755    0.749    
14    0.685    0.662    0.505    0.412    0.455    0.602    0.692    0.704    0.702    0.713    0.664    0.701    
15    0.644    0.563    0.422    0.459    0.527    0.662    0.738    0.733    0.722    0.601    0.639    0.656    
16    0.565    0.473    0.473    0.523    0.595    0.707    0.758    0.749    0.602    0.574    0.579    0.591    
17    0.474    0.519    0.557    0.589    0.651    0.732    0.777    0.623    0.585    0.511    0.501    0.480    
18    0.530    0.605    0.627    0.644    0.684    0.748    0.618    0.603    0.523    0.424    0.416    0.403    
19    0.633    0.683    0.682    0.679    0.704    0.489    0.594    0.554    0.458    0.371    0.381    0.499    
20    0.706    0.728    0.706    0.693    0.369    0.479    0.559    0.478    0.414    0.365    0.480    0.624    
21    0.749    0.749    0.724    0.320    0.370    0.460    0.496    0.410    0.417    0.463    0.589    0.707    
22    0.769    0.765    0.387    0.339    0.383    0.427    0.427    0.427    0.533    0.560    0.674    0.753    
23    0.784    0.487    0.415    0.375    0.393    0.351    0.456    0.574    0.640    0.661    0.729    0.767    
24    0.584    0.500    0.439    0.374    0.323    0.345    0.575    0.653    0.722    0.716    0.749    0.782     
""",
}


def parse_skill_by_lead(raw_results, init_months=None):
    """Parse lead values in the requested initialization-month order."""
    if init_months is None:
        init_months = INIT_MONTHS
    values_by_month = {month: {} for month in MONTH_CYCLE}
    for line in raw_results.strip().splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        lead = int(parts[0])
        for season, value in zip(RESULT_COLUMN_ORDER, parts[1:]):
            values_by_month[SEASON_TO_MONTH[season]][lead] = float(value)

    skill = np.full((len(init_months), MAX_LEAD), np.nan)
    for row_i, month in enumerate(init_months):
        for lead, value in values_by_month[month].items():
            if 1 <= lead <= MAX_LEAD:
                skill[row_i, lead - 1] = value
    return skill


def build_target_months(start_month=None):
    """Build target-month labels and calendar-year offsets from the start month."""
    if start_month is None:
        start_month = TARGET_START_MONTH
    n_targets = 12 + MAX_LEAD - 1
    start_idx = MONTH_CYCLE.index(start_month)
    target_months = []
    target_years = []
    for i in range(n_targets):
        m_idx = (start_idx + i) % 12
        year = (start_idx + i) // 12
        target_months.append(MONTH_CYCLE[m_idx])
        target_years.append(year)
    return target_months, target_years


def build_plot_matrix(skill_by_lead):
    n_targets = 12 + MAX_LEAD - 1
    z = np.full((12, n_targets), np.nan)
    for init_i in range(12):
        for lead in range(1, MAX_LEAD + 1):
            target_i = init_i + lead - 1
            if 0 <= target_i < n_targets:
                z[init_i, target_i] = skill_by_lead[init_i, lead - 1]
    return z


def build_hatch_mask(z):
    hatch_mask = np.zeros_like(z, dtype=bool)
    hatch_mask[~np.isnan(z) & (z < 0.5)] = True
    return hatch_mask


def plot_model(model_name):
    if model_name not in MODEL_TABLES:
        raise ValueError(f"Unknown model: {model_name}")

    target_months, target_years = build_target_months()
    skill_by_lead = parse_skill_by_lead(MODEL_TABLES[model_name])
    z = build_plot_matrix(skill_by_lead)
    hatch_mask = build_hatch_mask(z)
    n_targets = z.shape[1]

    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=300)

    cmap = LinearSegmentedColormap.from_list(
        "enso_red",
        ["#fff1ec", "#ff9b7f", "#fb5a49", "#b30017"]
    )
    cmap.set_bad("#cfcfcf")

    ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.3,
        vmax=0.95,
        extent=(-0.5, n_targets - 0.5, -0.5, 11.5)
    )

    for r in range(12):
        for c in range(n_targets):
            if hatch_mask[r, c]:
                rect = Rectangle(
                    (c - 0.5, r - 0.5),
                    1, 1,
                    facecolor=(1, 1, 1, 0),
                    edgecolor="0.35",
                    hatch="///",
                    linewidth=0.0
                )
                ax.add_patch(rect)

    for xpos in [11, 23]:
        ax.axvline(
            xpos,
            color="#1e9bd7",
            linestyle="--",
            linewidth=1.5
        )

    xtick_pos = np.arange(0, n_targets, 3)
    xtick_labels = [rf"{target_months[i]}$^{target_years[i]}$"
                    for i in xtick_pos]

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=12)
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(INIT_MONTHS, fontsize=12)

    ax.set_xlabel("Target month", fontsize=14, labelpad=2)
    ax.set_ylabel("Initial time", fontsize=14, labelpad=2)
    ax.set_title(
        f"Seasonality of Niño3.4 correlation skill ({model_name})",
        fontsize=14,
        pad=8
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=3,
        width=1.2,
        pad=3
    )

    ax.set_xlim(-0.5, n_targets - 0.5)
    ax.set_ylim(-0.5, 11.5)

    plt.tight_layout(pad=0.6)
    fig.savefig(f"nc_seasonal_{model_name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"nc_seasonal_{model_name}.pdf", bbox_inches="tight")
    plt.close(fig)
