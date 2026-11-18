"""Portable locations for the original figure calculations."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('TDA_DATA_DIR', str(ROOT / 'dataset'))).expanduser().resolve()
PREDICTION_ROOT = Path(os.environ.get('TDA_PREDICTIONS_DIR', str(ROOT / 'output/test'))).expanduser().resolve()
STAT_ROOT = Path(os.environ.get('TDA_STAT_DIR', str(ROOT / 'stat'))).expanduser().resolve()
SST_MEAN_FILE_PATH = Path(os.environ.get('TDA_SST_MEAN', str(ROOT / 'resources/sst_mean.npy'))).expanduser().resolve()
SST_STD_FILE_PATH = Path(os.environ.get('TDA_SST_STD', str(ROOT / 'resources/sst_std.npy'))).expanduser().resolve()
