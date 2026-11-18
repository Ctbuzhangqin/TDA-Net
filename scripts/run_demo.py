#!/usr/bin/env python3
"""Run one normalized monthly initialization through a released TDA-Net member."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/member1.bin')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/member1.json')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'example_data/GODAS')
    p.add_argument('--init', default='2021_12', help='Initialization YYYY_M; inputs must already be normalized.')
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--stat-root', type=Path, help='Optional directory with mean/sst.npy and std/sst.npy for Celsius output.')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'output/demo')
    p.add_argument('--check-inputs', action='store_true', help='Validate input files/configuration without loading weights or running the model.')
    return p.parse_args()

def main():
    args = parse_args()
    import numpy as np
    year, month = map(int, args.init.split('_'))
    if not 1 <= month <= 12 or args.steps < 1 or args.threads < 1:
        raise ValueError('Month must be 1..12; steps and threads must be positive.')
    cfg = json.loads(args.config.read_text())
    if cfg.get('in_steps', 1) != 1:
        raise ValueError('This demo expects the released one-month-input configuration.')
    shape = tuple(cfg['input_shape'])
    variables = cfg['var_list']
    if variables != ['so', 'thetao', 'tos', 'uo', 'vo', 'zos']:
        raise ValueError('Unexpected ocean variable order in configuration.')
    def load(var, channels):
        path = args.data_dir / var / ('%d_%d.npy' % (year, month))
        data = np.load(path, allow_pickle=False)
        if data.shape == shape and channels == 1:
            data = data[None]
        if data.shape != (channels,) + shape:
            raise ValueError('%s: expected %s, got %s' % (path, (channels,) + shape, data.shape))
        if not np.isfinite(data).all():
            raise ValueError('%s: input contains nonfinite values' % path)
        return data.astype(np.float32, copy=False)
    ocean = np.concatenate([load(v, c) for v, c in zip(variables, cfg['in_chans'])], axis=0)[None]
    atmo = np.concatenate([load(v, 1) for v in ['tauu', 'tauv']], axis=0)[None]
    print('Validated normalized ocean %s, atmosphere %s, month %d' % (ocean.shape, atmo.shape, month), flush=True)
    if args.check_inputs:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError('Download the matching weight first: %s. See README.md (Data and weights).' % args.checkpoint)
    import torch
    sys.path.insert(0, str(ROOT))
    from model import ORCADLConfig, ORCADLModel
    torch.set_num_threads(args.threads)
    torch.manual_seed(1)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu or a GPU machine.')
    start = time.perf_counter()
    print('Loading the trained model on %s ...' % device, flush=True)
    model = ORCADLModel(ORCADLConfig.from_json_file(str(args.config)))
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    if device == 'cuda':
        torch.cuda.synchronize()
    loaded = time.perf_counter()
    with torch.inference_mode():
        result = model(ocean_vars=torch.from_numpy(ocean).to(device),
                       atmo_vars=torch.from_numpy(atmo).to(device),
                       month=torch.tensor([month], dtype=torch.long, device=device),
                       predict_time_steps=args.steps, return_dict=True).preds
    if device == 'cuda':
        torch.cuda.synchronize()
    inferred = time.perf_counter()
    result = result.detach().cpu().numpy()
    if args.steps == 1:
        result = result[:, None]
    expected = (1, args.steps, sum(cfg['out_chans'])) + shape
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError('Invalid prediction: shape %s; expected %s; all values must be finite.' % (result.shape, expected))
    index = variables.index('tos')
    channel = sum(cfg['out_chans'][:index])
    sst = result[:, :, channel]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'tos_normalized.npy', sst)
    targets = []
    for lead in range(1, args.steps + 1):
        target_year, m0 = divmod(year * 12 + month - 1 + lead, 12)
        targets.append('%d_%d' % (target_year, m0 + 1))
    if args.stat_root:
        mean = np.load(args.stat_root / 'mean/sst.npy', allow_pickle=False)
        std = np.load(args.stat_root / 'std/sst.npy', allow_pickle=False)
        physical = np.stack([sst[:, i] * std[int(t.split('_')[1]) - 1] + mean[int(t.split('_')[1]) - 1]
                             for i, t in enumerate(targets)], axis=1)
        np.save(args.output_dir / 'tos_celsius.npy', physical)
    summary = {'initialization': args.init, 'target_months': targets, 'member': 'single trained member; not the ten-member paper ensemble',
               'shape': list(sst.shape), 'finite': bool(np.isfinite(sst).all()),
               'minimum': float(sst.min()), 'maximum': float(sst.max()), 'mean': float(sst.mean()),
               'device': device, 'cpu_threads': args.threads, 'python': platform.python_version(),
               'torch': torch.__version__, 'numpy': np.__version__,
               'load_seconds': loaded - start, 'inference_seconds': inferred - loaded}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Saved demo outputs to %s' % args.output_dir)

if __name__ == '__main__':
    main()
