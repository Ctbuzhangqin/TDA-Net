"""
Extract a Niño3.4 forecast for one initialization (lead months 1–24).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta


ROOT = Path(__file__).resolve().parents[1]
exp_name = 'tda_ensemble'
PRED_DIR = str(ROOT / 'output/test' / exp_name / 'preds')
INIT_TIMES_FILE = str(ROOT / 'output/test' / exp_name / 'init_times.json')
TRUE_DIR = str(ROOT / 'dataset/GODAS/tos')
NINO34_LAT_SLICE = slice(59, 69)
NINO34_LON_SLICE = slice(190, 241)
CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020
TARGET_INIT_TIME = '2014_8'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', default=TARGET_INIT_TIME, help='Initialization YYYY_M (default: 2014_8).')
    parser.add_argument('--pred-dir', type=Path, default=ROOT / 'output/test/tda_ensemble',
                        help='Directory containing preds/ and init_times.json.')
    parser.add_argument('--godas-dir', type=Path, default=ROOT / 'dataset/GODAS')
    parser.add_argument('--sst-mean', type=Path, default=ROOT / 'resources/sst_mean.npy')
    parser.add_argument('--sst-std', type=Path, default=ROOT / 'resources/sst_std.npy')
    parser.add_argument('--output', type=Path, help='Output text file; defaults to output/evaluation/EXPERIMENT_nino34_init_YYYY_M.txt.')
    args = parser.parse_args(argv)
    try:
        year, month = map(int, args.init.split('_'))
        datetime(year, month, 1)
    except (ValueError, TypeError):
        parser.error('--init must be a valid initialization month in YYYY_M format.')
    args.init = f'{year}_{month}'
    for name in ('pred_dir', 'godas_dir', 'sst_mean', 'sst_std'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output is None:
        args.output = ROOT / 'output/evaluation' / f'{args.pred_dir.name}_nino34_init_{args.init}.txt'
    else:
        args.output = args.output.expanduser().resolve()
    return args


def runmean(data, window=1):
    """计算滑动平均"""
    n = len(data)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        result[i] = np.nanmean(data[start:end])
    return result


def load_true_data(start_year=1980, end_year=2026):
    """加载真实值数据"""
    data_list = []
    time_list = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            filepath = f'{TRUE_DIR}/{year}_{month}.npy'
            try:
                data = np.load(filepath)
                data_list.append(data)
                time_list.append((year, month))
            except FileNotFoundError:
                break
        else:
            continue
        break
    return np.array(data_list), time_list


def denormalize_true_data(true_data, time_list, mean, std):
    """按真实值的实际月份反标准化，支持末年不足 12 个月。"""
    month_indices = np.array([month - 1 for _, month in time_list], dtype=int)
    return true_data * std[month_indices] + mean[month_indices]


def compute_climatology(true_data, time_list, start_year, end_year):
    """计算月气候态"""
    climatology = np.zeros((12, true_data.shape[1], true_data.shape[2]))
    counts = np.zeros(12)
    for i, (year, month) in enumerate(time_list):
        if start_year <= year <= end_year:
            climatology[month - 1] += true_data[i]
            counts[month - 1] += 1
    for m in range(12):
        if counts[m] > 0:
            climatology[m] /= counts[m]
    return climatology


def get_latitude_weights(lat_slice):
    """计算纬度权重"""
    start_idx = lat_slice.start
    stop_idx = lat_slice.stop
    n_lats = stop_idx - start_idx
    lats = np.linspace(-5, 5, n_lats)
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.sum()
    return weights.reshape(-1, 1)


def compute_nino34_index(data, climatology, time_list):
    """计算 Nino3.4 指数"""
    nino34_indices = []
    lat_weights = get_latitude_weights(NINO34_LAT_SLICE)
    for i, (_, month) in enumerate(time_list):
        anomaly = data[i] - climatology[month - 1]
        nino34_region = anomaly[NINO34_LAT_SLICE, NINO34_LON_SLICE]
        valid_mask = ~np.isnan(nino34_region)
        weighted_sum = np.nansum(nino34_region * lat_weights)
        weight_sum = np.sum(valid_mask.astype(float) * lat_weights)
        nino34_index = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        nino34_indices.append(nino34_index)
    return runmean(np.array(nino34_indices))


def load_pred_data_pt(lead):
    """加载预测数据"""
    filepath = f'{PRED_DIR}/tos_step{lead}.pt'
    pred_data = torch.load(filepath, weights_only=False)
    init_times = json.load(open(INIT_TIMES_FILE))
    pred_times = []
    for init_time in init_times:
        year, month = map(int, init_time.split('_'))
        init_date = datetime(year, month, 1)
        pred_date = init_date + relativedelta(months=lead)
        pred_times.append(f"{pred_date.year}-{pred_date.month:02d}")
    return pred_data, pred_times, init_times


def denormalize_pred_data(pred_data, pred_times, mean, std):
    """按预测目标月份对预测数据做反标准化"""
    pred_data_denorm = []
    for i, time_str in enumerate(pred_times):
        month = int(time_str.split('-')[1])
        pred_data_denorm.append(pred_data[i] * std[month - 1] + mean[month - 1])
    return np.array(pred_data_denorm)


def compute_pred_climatology(pred_data, pred_times, start_year, end_year):
    """计算预测数据的月气候态"""
    climatology = np.zeros((12, pred_data.shape[1], pred_data.shape[2]))
    counts = np.zeros(12)
    for i, time_str in enumerate(pred_times):
        year = int(time_str.split('-')[0])
        month = int(time_str.split('-')[1])
        if start_year <= year <= end_year:
            climatology[month - 1] += pred_data[i]
            counts[month - 1] += 1
    for m in range(12):
        if counts[m] > 0:
            climatology[m] /= counts[m]
    return climatology


def compute_pred_nino34_index(pred_data, climatology, pred_times):
    """计算预测的 Nino3.4 指数"""
    nino34_indices = []
    lat_weights = get_latitude_weights(NINO34_LAT_SLICE)
    for i, time_str in enumerate(pred_times):
        month = int(time_str.split('-')[1])
        anomaly = pred_data[i] - climatology[month - 1]
        nino34_region = anomaly[NINO34_LAT_SLICE, NINO34_LON_SLICE]
        valid_mask = ~np.isnan(nino34_region)
        weighted_sum = np.nansum(nino34_region * lat_weights)
        weight_sum = np.sum(valid_mask.astype(float) * lat_weights)
        nino34_index = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        nino34_indices.append(nino34_index)
    return runmean(np.array(nino34_indices))


def main():
    global exp_name, PRED_DIR, INIT_TIMES_FILE, TRUE_DIR, TARGET_INIT_TIME
    args = parse_args()
    exp_name = args.pred_dir.name
    PRED_DIR = str(args.pred_dir / 'preds')
    INIT_TIMES_FILE = str(args.pred_dir / 'init_times.json')
    TRUE_DIR = str(args.godas_dir / 'tos')
    TARGET_INIT_TIME = args.init
    print("=" * 60)
    print(f"提取起报时间 {TARGET_INIT_TIME} 的预测结果")
    print("=" * 60)

    # 1. 加载真实值数据
    print("\n[1] 加载真实值数据...")
    mean = np.load(args.sst_mean)
    std = np.load(args.sst_std)
    true_data, time_list = load_true_data()
    true_data = denormalize_true_data(true_data, time_list, mean, std)

    # 2. 计算气候态
    print(f"\n[2] 计算气候态 ({CLIM_START_YEAR}-{CLIM_END_YEAR})...")
    climatology = compute_climatology(true_data, time_list, CLIM_START_YEAR, CLIM_END_YEAR)

    # 3. 计算真实值的 Nino3.4 指数
    print("\n[3] 计算真实值 Nino3.4 指数...")
    true_nino34 = compute_nino34_index(true_data, climatology, time_list)

    # 创建时间到索引的映射
    time_to_idx = {}
    for i, (year, month) in enumerate(time_list):
        time_to_idx[(year, month)] = i

    # 4. 提取目标起报时间的预测结果
    print(f"\n[4] 提取起报时间 {TARGET_INIT_TIME} 的预测结果...")

    results = []
    trues = []
    preds = []

    for lead in range(1, 25):
        pred_data, pred_times, init_times = load_pred_data_pt(lead)
        pred_data = denormalize_pred_data(pred_data, pred_times, mean, std)
        # 找到目标起报时间的索引
        try:
            target_idx = init_times.index(TARGET_INIT_TIME)
        except ValueError:
            print(f"    警告: 起报时间 {TARGET_INIT_TIME} 不在 init_times 中")
            continue

        # 计算预测气候态
        pred_climatology = compute_pred_climatology(pred_data, pred_times, CLIM_START_YEAR, CLIM_END_YEAR)

        # 计算预测的 Nino3.4 指数
        pred_nino34 = compute_pred_nino34_index(pred_data, pred_climatology, pred_times)

        # 获取该起报时间对应的预测值
        pred_value = pred_nino34[target_idx]
        pred_time = pred_times[target_idx]

        # 获取对应的真实值
        year = int(pred_time.split('-')[0])
        month = int(pred_time.split('-')[1])

        if (year, month) in time_to_idx:
            true_idx = time_to_idx[(year, month)]
            true_value = true_nino34[true_idx]
        else:
            true_value = np.nan

        results.append({
            'lead': lead,
            'pred_time': pred_time,
            'pred_value': pred_value,
            'true_value': true_value
        })
        trues.append(true_value)
        preds.append(pred_value)
        print(f"    Lead {lead:2d}: {pred_time} | Pred={pred_value:7.4f} | True={true_value:7.4f}")
    print(trues)
    print(preds)
    # 5. 保存结果
    output_file = args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"起报时间: {TARGET_INIT_TIME}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Lead':<6} {'预测时间':<12} {'预测值':<12} {'真实值':<12}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['lead']:<6} {r['pred_time']:<12} {r['pred_value']:<12.4f} {r['true_value']:<12.4f}\n")

    print(f"\n结果已保存至 {output_file}")


if __name__ == '__main__':
    main()
