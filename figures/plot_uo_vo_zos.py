
from _paths import DATA_ROOT, PREDICTION_ROOT, STAT_ROOT, SST_MEAN_FILE_PATH, SST_STD_FILE_PATH
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ================= 基础配置 =================
EXP_NAMES = ['orca_member1', 'member1']  # ORCA-DL baseline and TDA-Net member1
LEAD = 24
EVAL_START = 2011
EVAL_END = 2023
TARGET_MONTHS = [3, 4, 5]  # MAM: March, April, May

TRUE_DIR = str(DATA_ROOT / 'GODAS')
STAT_DIR = str(STAT_ROOT)
PRED_DIR_TMPL = str(PREDICTION_ROOT / '{exp_name}/preds')
INIT_TIMES_FILE_TMPL = str(PREDICTION_ROOT / '{exp_name}/init_times.json')

VAR_MAPPING = {
    'uo': ('uo', 'ucur', True),
    'vo': ('vo', 'vcur', True),
    'zos': ('zos', 'sshg', False),
}

# ================= [新增] 空间相关系数计算模块 =================

def compute_latitude_weights(mask):
    """计算纬度余弦权重 (lat, lon)"""
    lat_size = mask.shape[0]
    lat = np.linspace(-63.5, 63.5, lat_size)
    cos_lat = np.cos(np.deg2rad(lat))
    lat_weights = cos_lat[:, np.newaxis] * mask.astype(np.float64)
    return lat_weights

def weighted_pearson_corr(pred, true, weights):
    """计算加权空间模式相关系数 (一维数组)"""
    # 过滤掉 NaN 值和无效权重区域 (如陆地)
    valid = ~(np.isnan(pred) | np.isnan(true)) & (weights > 0)
    if not np.any(valid):
        return np.nan
        
    p_valid, t_valid, w_valid = pred[valid], true[valid], weights[valid]
    
    # 加权平均
    p_mean = np.average(p_valid, weights=w_valid)
    t_mean = np.average(t_valid, weights=w_valid)
    
    # 异常值
    p_anom = p_valid - p_mean
    t_anom = t_valid - t_mean
    
    # 协方差与方差
    cov = np.sum(w_valid * p_anom * t_anom)
    var_p = np.sum(w_valid * p_anom**2)
    var_t = np.sum(w_valid * t_anom**2)
    
    if var_p == 0 or var_t == 0:
        return np.nan
    return cov / np.sqrt(var_p * var_t)

# ================= 数据读取与处理核心逻辑 =================

def load_land_sea_mask(stat_name, is_multilayer):
    mean = np.load(f'{STAT_DIR}/mean/{stat_name}.npy')
    if is_multilayer:
        ocean_mask = ~np.isnan(mean[0, 0])
    else:
        ocean_mask = ~np.isnan(mean[0])
    return ocean_mask

def get_true_djf_mean(var_name, true_var_name, stat_name, is_multilayer):
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
                denorm_data = data * std_stat[month - 1] + mean_stat[month - 1]
                if is_multilayer:
                    valid_data.append(denorm_data[0])
                else:
                    valid_data.append(denorm_data)
            except FileNotFoundError:
                continue
                
    if not valid_data:
        raise ValueError(f"没有找到 {true_var_name} 在指定期间的有效数据！")
        
    return np.nanmean(valid_data, axis=0)

def get_pred_djf_mean(exp_name, var_name, stat_name, is_multilayer):
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
            denorm_data = pred_data[i] * std_stat[p_month - 1] + mean_stat[p_month - 1]
            if is_multilayer:
                valid_data.append(denorm_data[0])
            else:
                valid_data.append(denorm_data)
                
    if not valid_data:
        raise ValueError(f"实验 {exp_name} 没有找到在指定期间的有效预测数据！")

    return np.nanmean(valid_data, axis=0)

# ================= 绘图与主函数 =================

def main():
    print("正在计算平均态数据...")
    
    ocean_mask = load_land_sea_mask(VAR_MAPPING['zos'][1], VAR_MAPPING['zos'][2])
    
    # [新增] 计算纬度权重矩阵
    lat_weights = compute_latitude_weights(ocean_mask)
    
    print("获取真实值 (GODAS)...")
    true_zos = get_true_djf_mean('zos', *VAR_MAPPING['zos'])
    true_uo = get_true_djf_mean('uo', *VAR_MAPPING['uo'])
    true_vo = get_true_djf_mean('vo', *VAR_MAPPING['vo'])
    
    # [新增] 提前拼接真实的流速场及其权重，为展平拼接法做准备
    true_curr_concat = np.concatenate([true_uo.flatten(), true_vo.flatten()])
    weights_curr_concat = np.concatenate([lat_weights.flatten(), lat_weights.flatten()])
    
    pred_data_dicts = []
    for exp_name in EXP_NAMES:
        print(f"获取预测值 ({exp_name})...")
        p_zos = get_pred_djf_mean(exp_name, 'zos', VAR_MAPPING['zos'][1], VAR_MAPPING['zos'][2])
        p_uo = get_pred_djf_mean(exp_name, 'uo', VAR_MAPPING['uo'][1], VAR_MAPPING['uo'][2])
        p_vo = get_pred_djf_mean(exp_name, 'vo', VAR_MAPPING['vo'][1], VAR_MAPPING['vo'][2])
        
        # [新增] 计算当前模型的 Corr_zos
        corr_zos = weighted_pearson_corr(p_zos.flatten(), true_zos.flatten(), lat_weights.flatten())
        
        # [新增] 展平拼接法计算当前模型的 Corr_curr
        pred_curr_concat = np.concatenate([p_uo.flatten(), p_vo.flatten()])
        corr_curr = weighted_pearson_corr(pred_curr_concat, true_curr_concat, weights_curr_concat)
        
        print(f"-> {exp_name} | Corr_zos: {corr_zos:.2f}, Corr_curr: {corr_curr:.2f}")
        
        pred_data_dicts.append({
            "name": exp_name, "zos": p_zos, "uo": p_uo, "vo": p_vo,
            "corr_zos": corr_zos, "corr_curr": corr_curr
        })

    def apply_mask(field):
        return np.where(ocean_mask, field, np.nan)

    # [修改] 在 Title 中嵌入计算出的相关系数结果
    plot_configs = [
        {"title": f"GODAS", 
         "zos": apply_mask(true_zos), "uo": apply_mask(true_uo), "vo": apply_mask(true_vo)},
         
        {"title": f"ORCA-DL (Lead={LEAD}) | $\\mathrm{{Corr}}_{{\\mathrm{{zos}}}}$={pred_data_dicts[0]['corr_zos']:.2f}, $\\mathrm{{Corr}}_{{\\mathrm{{curr}}}}$={pred_data_dicts[0]['corr_curr']:.2f}", 
         "zos": apply_mask(pred_data_dicts[0]['zos']), "uo": apply_mask(pred_data_dicts[0]['uo']), "vo": apply_mask(pred_data_dicts[0]['vo'])},
         
        {"title": f"TDA-Net (Lead={LEAD}) | $\\mathrm{{Corr}}_{{\\mathrm{{zos}}}}$={pred_data_dicts[1]['corr_zos']:.2f}, $\\mathrm{{Corr}}_{{\\mathrm{{curr}}}}$={pred_data_dicts[1]['corr_curr']:.2f}", 
         "zos": apply_mask(pred_data_dicts[1]['zos']), "uo": apply_mask(pred_data_dicts[1]['uo']), "vo": apply_mask(pred_data_dicts[1]['vo'])},
    ]

    print("开始绘图...")
    
    lat = np.linspace(-63.5, 63.5, 128)
    lon = np.linspace(0.5, 359.5, 360)
    LON, LAT = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 17))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.12, hspace=0.2)
    
    all_zos = np.concatenate([c['zos'].flatten() for c in plot_configs])
    vmin, vmax = np.nanpercentile(all_zos, 1), np.nanpercentile(all_zos, 99)

    skip = (slice(None, None, 5), slice(None, None, 8))

    # 定义坐标轴刻度及格式化标签
    xticks = [0, 60, 120, 180, 240, 300, 360]
    xticklabels = ['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°']
    
    yticks = [-60, -30, 0, 30, 60]
    yticklabels = ['60°S', '30°S', '0°', '30°N', '60°N']

    for ax, cfg in zip(axes, plot_configs):
        ax.set_facecolor('lightgray')
        cf = ax.contourf(LON, LAT, cfg['zos'], levels=30, cmap='RdYlBu_r', 
                         vmin=vmin, vmax=vmax, extend='both')
        
        q = ax.quiver(LON[skip], LAT[skip], cfg['uo'][skip], cfg['vo'][skip],
                      scale=8, width=0.003, headwidth=4, headlength=5, color='black')
        
        ax.set_title(cfg['title'], fontsize=18, pad=10)
        ax.set_xlim([0, 360])
        ax.set_ylim([-63.5, 63.5])
        
        # ---------------- 核心修改点：应用自定义经纬度刻度 ----------------
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=16)
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=16)
        # ------------------------------------------------------------------
        
    # ================= 添加底部公共图例 =================
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.01])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Sea Surface Height (m)', fontsize=16)

    axes[-1].quiverkey(q, X=0.90, Y=-0.15, U=0.5, 
                       label='0.5 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 16})

    save_path = "compare_djf_mean_state.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"绘图完成！已保存为: {save_path}")

if __name__ == '__main__':
    main()