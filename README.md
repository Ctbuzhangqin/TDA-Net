# Long-lead ENSO prediction across the spring predictability barrier


## Installation

Linux; Python 3.9 for training/inference and Python 3.8 for figures.
Training defaults to four NVIDIA GPUs; the demo supports CPU.

```bash
conda create -n tda-net python=3.9.17
conda activate tda-net
python -m pip install -r requirements.txt

conda create -n tda-net-plots python=3.8.13
conda activate tda-net-plots
python -m pip install -r requirements-plot.txt
```

Run all commands below from the repository root.

## Data and weights

Download [trained weights](https://huggingface.co/HouZhongYi/TDA-Net) and
[GODAS demo data/statistics](https://huggingface.co/datasets/HouZhongYi/TDA-Net-data).
From the repository root:

```bash
hf download HouZhongYi/TDA-Net --include "checkpoints/member1.bin" --local-dir .
hf download HouZhongYi/TDA-Net-data --repo-type dataset --local-dir . \
  --include "example_data/*" "stat/*" "resources/*"
# Optional: download all ten ensemble members.
hf download HouZhongYi/TDA-Net --include "checkpoints/*.bin" --local-dir .
```

The minimal demo needs [member1](https://huggingface.co/HouZhongYi/TDA-Net/resolve/main/checkpoints/member1.bin) (3.65 GB), eight normalized December 2021 GODAS
inputs (25.1 MB), and monthly statistics. Downloads follow this layout:

```text
checkpoints/member1.bin ... member10.bin
example_data/GODAS/VARIABLE/2021_12.npy
stat/{mean,std}/STAT_VARIABLE.npy
resources/{sst_mean,sst_std}.npy
```

Ocean input order: `so thetao tos uo vo zos`; atmosphere: `tauu tauv`.
Statistics use `salt pottmp sst ucur vcur sshg uflx vflx`, respectively.
Inputs have a 128 × 360 grid; `so thetao uo vo` each have 16 levels.
The supplied inputs are already interpolated and normalized; **do not normalize
again**. Matching member configurations are included under `configs/`.

## Quick demo

Open [demo.ipynb](demo.ipynb) and run the cells in order:

```bash
conda activate tda-net
python -m pip install notebook
jupyter notebook demo.ipynb
```

The notebook loads data/statistics and weights, predicts January 2022 from
December 2021, restores physical units, and plots SST. Outputs are saved in
`output/notebook_demo/`: `tos_normalized.npy`, `tos_celsius.npy`, `summary.json`
and `sst_demo.png`. Default SST shape: `1 × 1 × 128 × 360`.
Set `STEPS = 6` for multiple leads, `MEMBERS = list(range(1, 11))` for the
ensemble, or add months to `INIT_TIMES` for a batch (supply those input files).

For a command-line demo:

```bash
python scripts/run_demo.py --check-inputs
python scripts/run_demo.py --device cpu --stat-root stat
```

This saves SST and a summary in `output/demo/`; reference output is in
[examples/reference/](examples/reference/). A four-thread CPU run took about
13 seconds to load member1 and 2 seconds to predict. Use `--device cuda` for GPU,
`--steps` for multiple leads, or `--checkpoint` with the matching `--config`.

## Training and prediction

```bash
conda activate tda-net
bash train.sh 1
python scripts/predict_ensemble.py --data-dir dataset/GODAS \
  --vars so thetao tos uo vo zos --dry-run
```

Full training/test datasets are separate from the single-month demo:
`dataset/train/CMIP_MODEL/VARIABLE/YYYY_M.npy`,
`dataset/val/{SODA2,ORAS5}/VARIABLE/YYYY_M.npy` and
`dataset/GODAS/VARIABLE/YYYY_M.npy`.

`python train.py` without arguments displays help.
`train.sh` takes a random seed (default 1). Its validation data are ORAS5/SODA
through 1979 inclusive; training uses the supplied CMIP years. Paths can be
overridden with `TRAIN_DATA_DIR`, `SODA_DATA_DIR`, `ORAS5_DATA_DIR` and `OUTPUT_DIR`.

Remove `--dry-run` to predict with members 1–10 and save their average to
`output/test/tda_ensemble`. Default initialization years are 2011–2023;
change them with `--init-start-year` and `--init-end-year`.

## Evaluation and figures

For the 1991–2020 climatology below, first generate ensemble predictions using
`--init-start-year 1988`; also supply GODAS observations for 1980–2023 and the
SST statistics. Test scores use forecast target years 2011–2023.

```bash
conda activate tda-net
python scripts/evaluate_nino.py --pred-dir output/test/tda_ensemble \
  --clim-start 1991 --clim-end 2020
# Extract a single initialization:
python evaluation/extract_nino_forecast.py --init 2014_8

conda activate tda-net-plots
python scripts/run_figure.py fig1
python scripts/run_figure.py --help
```

The extractor writes `output/evaluation/tda_ensemble_nino34_init_2014_8.txt`;
use `--pred-dir` or `--output` to select other forecast inputs or an output path.

Figures are saved under `figures/`. Field plots need forecasts in
`output/test/EXPERIMENT/{init_times.json,preds/}`, GODAS fields and statistics;
use `tda_ensemble` for TDA-Net and `ensemble_10_orca` for the comparison.
For external locations, set `TDA_DATA_DIR`, `TDA_PREDICTIONS_DIR`, `TDA_STAT_DIR`,
`TDA_SST_MEAN` and `TDA_SST_STD`.

## GODAS preprocessing

For your own GODAS input, install CDO and wget, place downloaded statistics under
`stat/`, then regrid and normalize the selected years:

```bash
conda activate tda-net
bash preprocessing/download_process.sh 2021
python preprocessing/convert_processed_nc_to_godas_npy.py --years 2021
```

The shell script interpolates onto the model grid; the Python converter changes
temperature from K to °C and salinity from kg/kg to g/kg before normalization.
Current, sea level and wind stress use m/s, m and N/m². Normalization uses the
initialization month; the notebook restores forecasts using each target month.

## License

Original TDA-Net contributions are made available under the MIT License. See [NOTICE](NOTICE) for third-party code and license details.
