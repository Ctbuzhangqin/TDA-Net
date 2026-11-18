#!/usr/bin/env python3
"""Run index metrics on GODAS for 2011-2023 by default, with an explicit climatology period."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['overall', 'seasonal'], default='overall')
    p.add_argument('--pred-dir', type=Path, default=ROOT / 'output/test/tda_ensemble', help='Experiment directory containing preds/ and init_times.json. Default: output/test/tda_ensemble.')
    p.add_argument('--godas-dir', type=Path, default=ROOT / 'dataset/GODAS')
    p.add_argument('--sst-mean', type=Path, default=ROOT / 'resources/sst_mean.npy')
    p.add_argument('--sst-std', type=Path, default=ROOT / 'resources/sst_std.npy')
    p.add_argument('--eval-start', type=int, default=2011, help='Inclusive forecast target start year (default: 2011).')
    p.add_argument('--eval-end', type=int, default=2023, help='Inclusive forecast target end year (default: 2023).')
    p.add_argument('--clim-start', type=int, required=True)
    p.add_argument('--clim-end', type=int, required=True)
    p.add_argument('--region', choices=['nino34','nino3','nino4'], default='nino34')
    p.add_argument('--leads', type=int, default=24)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'output/evaluation')
    args = p.parse_args()
    if args.eval_start > args.eval_end or args.clim_start > args.clim_end or args.leads < 1:
        p.error('Periods must be ordered and leads must be positive.')
    paths = {k: getattr(args, k).expanduser().resolve() for k in ['pred_dir','godas_dir','sst_mean','sst_std','output_dir']}
    for name in ['sst_mean','sst_std']:
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    for name in ['init_times.json'] + ['preds/tos_step%d.pt' % lead for lead in range(1, args.leads + 1)]:
        if not (paths['pred_dir'] / name).is_file():
            raise FileNotFoundError(paths['pred_dir'] / name)
    # The original main routine assumes whole observed calendar years.
    last_year = max(args.clim_end, args.eval_end)
    for year in range(1980, last_year + 1):
        for month in range(1,13):
            f = paths['godas_dir'] / 'tos' / ('%d_%d.npy' % (year,month))
            if not f.is_file():
                raise FileNotFoundError('Original metric code requires complete observed years: %s' % f)
    name = 'corr_pt_all' if args.mode == 'overall' else 'corr_pt_seasonal'
    spec = importlib.util.spec_from_file_location('tda_metric', ROOT / 'evaluation' / (name + '.py'))
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    metric.exp_name = paths['pred_dir'].name
    metric.PRED_DIR = str(paths['pred_dir'] / 'preds')
    metric.INIT_TIMES_FILE = str(paths['pred_dir'] / 'init_times.json')
    metric.TRUE_DIR = str(paths['godas_dir'] / 'tos')
    metric.SST_MEAN_FILE, metric.SST_STD_FILE = str(paths['sst_mean']), str(paths['sst_std'])
    metric.EVAL_START, metric.EVAL_END = args.eval_start, args.eval_end
    metric.CLIM_START_YEAR, metric.CLIM_END_YEAR = args.clim_start, args.clim_end
    metric.NUM_LEADS = args.leads
    metric.NINO34_LON_SLICE = {'nino34':slice(190,241),'nino3':slice(210,271),'nino4':slice(160,211)}[args.region]
    original_load = metric.load_true_data
    metric.load_true_data = lambda: original_load(start_year=1980, end_year=last_year)
    paths['output_dir'].mkdir(parents=True, exist_ok=True)
    configuration = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (paths['output_dir'] / 'evaluation_settings.json').write_text(json.dumps(configuration, indent=2) + '\n')
    os.chdir(paths['output_dir'])
    metric.main()

if __name__ == '__main__':
    main()
