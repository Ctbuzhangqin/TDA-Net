#!/usr/bin/env python3
"""Predict one member at a time, then average aligned saved forecasts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--init-start-year', type=int, default=2011, help='Inclusive initialization start year (default: 2011).')
    p.add_argument('--init-end-year', type=int, default=2023, help='Inclusive initialization end year (default: 2023).')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--vars', nargs='+', default=['tos'], choices=['so','thetao','tos','uo','vo','zos'])
    p.add_argument('--manifest', type=Path, default=ROOT / 'configs/ensemble.json')
    p.add_argument('--output-root', type=Path, default=ROOT / 'output/test')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.init_end_year < args.init_start_year or args.steps < 2:
        p.error('End year must be >= start year; use the demo for a single forecast step.')
    manifest = json.loads(args.manifest.read_text())
    experiments, commands = [], []
    for member in manifest['members']:
        checkpoint = ROOT / member['checkpoint']
        config = ROOT / member['config']
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError('Missing released weight: %s' % checkpoint)
        name = 'member%d' % member['member_id']
        experiments.append(name)
        output = args.output_root.resolve() / name
        cmd = [sys.executable, str(ROOT / 'predict_all.py'), '--save_preds', 'True', '--save_vars'] + args.vars
        cmd += ['--ckpt_list', str(checkpoint), '--config_path_list', str(config), '--data_dir', str(args.data_dir.resolve()),
                '--input_steps', '1', '--predict_steps', str(args.steps), '--output_dir', str(output), '--report_to', 'none',
                '--seed', '1', '--dataloader_num_workers', '0', '--per_device_eval_batch_size', str(args.batch_size),
                '--init_start_year', str(args.init_start_year), '--init_end_year', str(args.init_end_year)]
        commands.append(cmd)
    cmd = [sys.executable, str(ROOT / 'ensemble.py'), '--root', str(args.output_root.resolve()), '--experiments'] + experiments
    cmd += ['--weights'] + [str(m['weight']) for m in manifest['members']]
    cmd += ['--vars'] + args.vars + ['--output_dir', str(args.output_root.resolve() / manifest['experiment_name'])]
    commands.append(cmd)
    import shlex
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
