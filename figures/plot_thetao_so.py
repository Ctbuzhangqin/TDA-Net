
from _paths import DATA_ROOT, PREDICTION_ROOT, STAT_ROOT, SST_MEAN_FILE_PATH, SST_STD_FILE_PATH
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ================= 基础配置 =================
EXP_NAMES = ['orca_member1', 'member1']
LEAD = 24
EVAL_START = 2011
EVAL_END = 2023
TARGET_MONTHS = [3, 4, 5]  # MAM: March, April, May

TRUE_DIR = str(DATA_ROOT / 'GODAS')
STAT_DIR = str(STAT_ROOT)
PRED_DIR_TMPL = str(PREDICTION_ROOT / '{exp_name}/preds')
INIT_TIMES_FILE_TMPL = str(PREDICTION_ROOT / '{exp_name}/init_times.json')

VAR_MAPPING = {
    'thetao': ('thetao', 'pottmp'),
    'so': ('so', 'salt'),
}

DEPTHS = [10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# ================= 海水势密度与指标计算 =================

def calc_sigma0(S, T):
    """基于 EOS-80 计算势密度异常"""
    rho_w = (999.842594 + 6.793952e-2 * T - 9.095290e-3 * T**2 + 
             1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536332e-9 * T**5)
    A = 0.824493 - 4.0899e-3 * T + 7.6438e-5 * T**2 - 8.2467e-7 * T**3 + 5.3875e-9 * T**4
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2
    C = 4.8314e-4
    rho = rho_w + A * S + B * S**1.5 + C * S**2
    return rho - 1000.0

def get_true_djf_mean_all_layers(var_name, true_var_name, stat_name):
    mean_stat = np.load(f'{STAT_DIR}/mean/{stat_name}.npy')
    std_stat = np.load(f'{STAT_DIR}/std/{stat_name}.npy')
    true_dir = f'{TRUE_DIR}/{true_var_name}'
    
    valid_data = []
    for year in range(EVAL_START - 1, EVAL_END + 1):
        for month in range(1, 13):
            if not (EVAL_START <= year <= EVAL_END) or month not in TARGET_MONTHS:
                continue
                
            filepath = f'{true_dir}/{year}_{month}.npy'
            try:
                data = np.load(filepath)
                # 反标准化全部 16 层
                denorm_data = data * std_stat[month - 1] + mean_stat[month - 1]
                valid_data.append(denorm_data)
            except FileNotFoundError:
                continue
                
    if not valid_data:
        raise ValueError(f"没有找到 {true_var_name} 在指定期间的有效数据！")
        
    return np.nanmean(valid_data, axis=0) # Shape: (16, lat, lon)

def get_pred_djf_mean_all_layers(exp_name, var_name, stat_name):
    pred_dir = PRED_DIR_TMPL.format(exp_name=exp_name)
    init_file = INIT_TIMES_FILE_TMPL.format(exp_name=exp_name)
    
    filepath = f'{pred_dir}/{var_name}_step{LEAD}.pt'
    pred_data = torch.load(filepath, weights_only=False)
    if isinstance(pred_data, torch.Tensor):
        pred_data = pred_data.numpy()
        
    init_times = json.load(open(init_file))
    mean_stat = np.load(f'{STAT_DIR}/mean/{stat_name}.npy')
    std_stat = np.load(f'{STAT_DIR}/std/{stat_name}.npy')
    
    valid_data = []
    for i, init_time in enumerate(init_times):
        y, m = map(int, init_time.split('_'))
        init_date = datetime(y, m, 1)
        pred_date = init_date + relativedelta(months=LEAD)
        p_year, p_month = pred_date.year, pred_date.month
        
        if EVAL_START <= p_year <= EVAL_END and p_month in TARGET_MONTHS:
            # 反标准化全部 16 层
            denorm_data = pred_data[i] * std_stat[p_month - 1] + mean_stat[p_month - 1]
            valid_data.append(denorm_data)
                
    if not valid_data:
        raise ValueError(f"实验 {exp_name} 没有找到在指定期间的有效预测数据！")

    return np.nanmean(valid_data, axis=0) # Shape: (16, lat, lon)

def calc_ts_distribution_corr(s_true, t_true, s_pred, t_pred):
    """计算 2D T-S 联合概率分布 (直方图) 的 Pearson 相关系数"""
    # 设定固定的二维 bin 范围，确保两个矩阵网格完全对齐
    bins = [100, 100]
    range_lims = [[30, 40], [-2, 32]] # [盐度范围, 温度范围]
    
    H_true, _, _ = np.histogram2d(s_true, t_true, bins=bins, range=range_lims)
    H_pred, _, _ = np.histogram2d(s_pred, t_pred, bins=bins, range=range_lims)
    
    # 展平后计算 Pearson 相关系数
    corr = np.corrcoef(H_true.flatten(), H_pred.flatten())[0, 1]
    return corr

def calc_density_rmse(s_true, t_true, s_pred, t_pred):
    """计算势密度的空间均方根误差 (RMSE)"""
    sigma_true = calc_sigma0(s_true, t_true)
    sigma_pred = calc_sigma0(s_pred, t_pred)
    rmse = np.sqrt(np.nanmean((sigma_pred - sigma_true)**2))
    return rmse

# ... (此处省略与上文完全一致的 get_true_djf_mean_all_layers 和 get_pred_djf_mean_all_layers 数据读取函数) ...

def extract_valid_ts_points(T_data, S_data, subsample_step=1):
    """提取有效的 T-S 点对 (计算指标时不需要抽样，绘图时再抽样即可，保证指标精度)"""
    all_t, all_s, all_d = [], [], []
    num_layers = T_data.shape[0]
    
    for i in range(num_layers):
        valid_mask = ~(np.isnan(T_data[i]) | np.isnan(S_data[i]))
        t_valid = T_data[i][valid_mask]
        s_valid = S_data[i][valid_mask]
        d_valid = np.full_like(t_valid, DEPTHS[i])
        
        all_t.append(t_valid)
        all_s.append(s_valid)
        all_d.append(d_valid)
        
    all_t = np.concatenate(all_t)
    all_s = np.concatenate(all_s)
    all_d = np.concatenate(all_d)
    
    return all_t, all_s, all_d


def build_contour_label_positions(contour_set, xlim, min_vertices=12):
    """
    为每个等值线 level 选一个更稳定的标签位置，避免默认 clabel 因线段太短或太靠边而漏标。
    """
    x_targets = np.linspace(xlim[0] + 1.0, xlim[1] - 1.0, max(len(contour_set.levels), 1))
    manual_positions = []

    for level_idx, segments in enumerate(contour_set.allsegs):
        best_segment = None
        best_score = -1

        for seg in segments:
            if len(seg) < min_vertices:
                continue
            x_span = seg[:, 0].max() - seg[:, 0].min()
            y_span = seg[:, 1].max() - seg[:, 1].min()
            score = x_span + 0.3 * y_span
            if score > best_score:
                best_score = score
                best_segment = seg

        if best_segment is None:
            continue

        target_x = x_targets[min(level_idx, len(x_targets) - 1)]
        point_idx = np.argmin(np.abs(best_segment[:, 0] - target_x))
        manual_positions.append(tuple(best_segment[point_idx]))

    return manual_positions

# ================= 绘图与主函数 =================

def main():
    print("正在计算多层 T & S 平均态数据...")
    
    print("-> 加载 GODAS...")
    true_t = get_true_djf_mean_all_layers('thetao', *VAR_MAPPING['thetao'])
    true_s = get_true_djf_mean_all_layers('so', *VAR_MAPPING['so'])
    # 提取完整的不抽样点云用于计算高精度指标
    t_gt_full, s_gt_full, d_gt_full = extract_valid_ts_points(true_t, true_s)
    
    plot_data = [{
        "title": "GODAS", 
        "t": t_gt_full, "s": s_gt_full, "d": d_gt_full
    }]
    
    for exp_name in EXP_NAMES:
        print(f"-> 加载预测值 ({exp_name}) 并计算指标...")
        pred_t = get_pred_djf_mean_all_layers(exp_name, 'thetao', VAR_MAPPING['thetao'][1])
        pred_s = get_pred_djf_mean_all_layers(exp_name, 'so', VAR_MAPPING['so'][1])
        t_p_full, s_p_full, d_p_full = extract_valid_ts_points(pred_t, pred_s)
        
        # 1. 计算 2D 直方图联合分布相关系数
        corr_ts = calc_ts_distribution_corr(s_gt_full, t_gt_full, s_p_full, t_p_full)
        
        # 2. 计算密度的均方根误差
        rmse_sigma = calc_density_rmse(s_gt_full, t_gt_full, s_p_full, t_p_full)
        
        title = f"{exp_name}; Temp & Salt\n$Corr_{{T-S}}$={corr_ts:.2f}, $RMSE_{{\sigma_0}}$={rmse_sigma:.2f}"
        print(title)
        if exp_name == 'orca_member1':
            title = "ORCA-DL"
        else:
            title = "TDA-Net"
        plot_data.append({
            "title": title, 
            "t": t_p_full, "s": s_p_full, "d": d_p_full
        })

    print("开始绘制温盐(T-S)散点图...")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.25)
    
    s_grid, t_grid = np.meshgrid(np.linspace(30, 40, 100), np.linspace(-2, 32, 100))
    sigma_grid = calc_sigma0(s_grid, t_grid)
    sigma_levels = [20, 22, 24, 26, 28, 30]

    # 为了渲染速度，在这里仅对画图的点云进行抽样 (每 10 个点画 1 个)
    subsample = 10 

    for ax, data in zip(axes, plot_data):
        xlim = [30.5, 39.5]
        ylim = [-2, 32]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # cs = ax.contour(
        #     s_grid, t_grid, sigma_grid,
        #     levels=sigma_levels,
        #     colors='black',
        #     linestyles='dashed',
        #     alpha=0.8
        # )

        # manual_positions = build_contour_label_positions(cs, xlim=xlim)
        # if manual_positions:
        #     ax.clabel(
        #         cs,
        #         inline=True,
        #         fontsize=10,
        #         fmt='%1.0f',
        #         manual=manual_positions,
        #         use_clabeltext=True
        #     )
        # else:
        #     ax.clabel(
        #         cs,
        #         inline=True,
        #         fontsize=10,
        #         fmt='%1.0f',
        #         use_clabeltext=True
        #     )

        sc = ax.scatter(
            data['s'][::subsample], data['t'][::subsample],
            c=data['d'][::subsample], cmap='Spectral',
            s=1, alpha=0.5, vmin=0, vmax=1000
        )

        ax.set_title(data['title'], fontsize=14, pad=8)
        ax.set_xlabel('Salinity (g/kg)', fontsize=12)
        ax.set_ylabel('Temp (°C)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', labelsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Depth (m)', fontsize=12)
    cbar.ax.invert_yaxis()
    save_path = "ts_scatter_with_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"绘图完成！已保存为: {save_path}")

if __name__ == '__main__':
    # 注意把前面的数据提取函数补充完整再运行
    main()
    pass

# ================= 数据读取与处理核心逻辑 =================
