# SatPV-Field

Code release for source-supported regional photovoltaic (PV) generation-field forecasting and regional-to-local station forecasting.

The repository implements the four-stage pipeline used in the manuscript:

1. preprocess Himawari-8 observations and construct clear-sky-aligned samples;
2. forecast hourly irradiance guidance with a Ratio3DCNN and residual ConvLSTM;
3. construct target-excluded PV generation fields and forecast eight 15-min field horizons;
4. extract target-local field patches and forecast held-out station output.

The data, cached tensors, trained weights, and experimental outputs are **not included**. They may contain large or restricted operational data. This repository contains source code, launchers, evaluation code, and figure-generation code only.

## Reproducibility status

The authoritative paper configuration is:

- 496 PV stations in total;
- 446 source stations and 50 held-out target stations;
- 12 historical 15-min PV frames (`T-3 h` to `T-15 min`);
- 8 future 15-min horizons (`T` to `T+1 h 45 min`, corresponding to 15--120 min ahead of the latest observation);
- a 64 x 64 source-supported regional field;
- four observed and three forecast hourly irradiance anchors;
- Stage 3 `last_only + warp_direct_gate + full_swr_future`;
- Stage 4 5 x 5 predicted-field patches with a binary support-mask channel.

The final paper entry points are listed in [Paper-to-code map](docs/PAPER_CODE_MAP.md). Internal implementation files are retained because the wrappers import them, but the documented wrappers and commands below should be treated as authoritative.

## Repository layout

```text
SatPV-Field/
├── stage_1/        H8 preprocessing, indices, frame/sample/clear-sky caches
├── stage_2/        irradiance-guidance models and ablations
├── stage_3/        field construction, field forecasting, ablations
├── stage_4/        station forecasting, baselines, condition analyses
├── experiments/    common-intersection station evaluation
├── paper_support/  representation checks and matched graph comparison
├── Framework_draw/ manuscript figure-generation scripts
├── scripts/        release validation and environment checks
├── configs/        environment-variable template
└── docs/           data layout and paper-to-code documentation
```

Generated directories such as `data/`, `runs/`, and `outputs/` are ignored by Git.

## Environment

### Recommended container

The experiments were developed for Linux with a CUDA-enabled PyTorch image. The environment used during the final experiments was based on:

```text
pytorch:2.11.0-cuda13.0-cudnn9-devel
```

PyTorch is intentionally not pinned in `requirements.txt`: installing a generic wheel can replace the CUDA build supplied by the image. Confirm that the base interpreter can import CUDA-enabled PyTorch before installing the remaining packages.

### Virtual environment

On Debian/Ubuntu containers, install venv support if it is missing:

```bash
apt-get update
apt-get install -y python3-venv
```

Then run:

```bash
cd /path/to/SatPV-Field
bash setup_env_tuna.sh
source .venv/bin/activate
```

The setup script uses the Tsinghua PyPI mirror and inherits the container's PyTorch through `--system-site-packages`. To install manually with the same mirror:

```bash
python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --prefer-binary
```

Verify the environment:

```bash
python scripts/check_environment.py
```

## Data configuration

Copy the template and edit only the right-hand-side paths:

```bash
cp configs/env.example.sh configs/env.local.sh
source configs/env.local.sh
```

The important variables are:

| Variable | Purpose |
|---|---|
| `SATPV_PROJECT_ROOT` | repository root |
| `SATPV_DATA_ROOT` | external root for generated caches, checkpoints, and existing results |
| `H8_RAW_ROOT` | raw Himawari-8 NetCDF tree (`**/*.nc`) |
| `PV_ROOT` | parent of the train and validation/test station CSV directories |
| `NWP_ROOT` | parent of `old_NWP` metadata used for the regional NWP cache |
| `STATION_SPLIT_CSV` | full station table with at least `station_sn`, `lon`, and `lat` |
| `PYTHON` | Python executable used by shell launchers; defaults to `python` |

`OLD_DATA_ROOT` remains accepted as a backward-compatible alias for `SATPV_DATA_ROOT`.

The expected directory and column conventions are documented in [Data layout](docs/DATA_LAYOUT.md). No data file is copied into this repository.

## Quick validation without data

Static validation does not train a model and does not require the dataset:

```bash
python -m compileall -q stage_1 stage_2 stage_3 stage_4 \
  experiments paper_support Framework_draw scripts
python scripts/check_environment.py --code-only
```

After the environment variables and data are available, run:

```bash
python scripts/check_environment.py --strict
python stage_3/00_preflight.py --require_swr_checkpoint 0
```

## End-to-end pipeline

All commands below are run from the repository root. They do not require the repository to have a particular folder name.

### 1. Build H8 samples and clear-sky caches

```bash
bash stage_1/run_stage1_build_all.sh
```

This creates the processed center/big-patch NetCDF files, split indices, frame caches, four-history/three-future sample caches, and clear-sky cache under:

```text
${SATPV_DATA_ROOT}/stage1/data/
```

The raw NetCDF files are read recursively from `H8_RAW_ROOT`.

### 2. Train future irradiance guidance

Train the 5 x 5 patch-augmented Ratio3DCNN with the `conv3d_decoder` temporal head:

```bash
bash stage_2/run_train_3dcnn_ratio_big_patch_5x5.sh
```

Train the residual ConvLSTM correction used by the downstream Stage 3 configuration:

```bash
python stage_2/train_residual_convlstm_v2_no_center_weight.py
```

The downstream checkpoint is expected at:

```text
${SATPV_DATA_ROOT}/stage2/runs/h8_swr_residual_convlstm_on_bigpatch_conv3d_decoder_4h3f/
  v2_no_center_weight/best.pt
```

### 3. Construct and forecast source-supported PV fields

Build the leakage-safe station split, multisource cache, target selection, PV-field cache, and future-SWR anchor cache:

```bash
bash stage_3/run_stage3_build_caches.sh
```

The first step imports `STATION_SPLIT_CSV`. Fifty targets are then selected and removed before the field cache is constructed. Target PV measurements are not used by the regional branch.

Train only the final paper field forecaster:

```bash
python stage_3/ablation_motion_branch/train_motion_branch.py \
  --motion_branch warp_direct_gate \
  --temporal_collapse last_only \
  --swr_dependency full_swr_future \
  --run_root "${SATPV_DATA_ROOT}/stage3/runs/ablation_motion_branch_last_only"
```

The expected checkpoint is:

```text
${SATPV_DATA_ROOT}/stage3/runs/ablation_motion_branch_last_only/
  warp_direct_gate/swr_anchor_motion_fusion_pv_nwp_motion_warp_direct_gate_best.pt
```

To reproduce all motion-branch variants instead:

```bash
bash stage_3/ablation_motion_branch_last_only/run_train_motion_branch_last_only_ablation.sh
```

### 4. Train regional-to-local station forecasting

```bash
bash stage_4/run_stage4_train_per_station_stage3_field_patch.sh
```

This trains one model per held-out target station. Each model receives target history, future station conditions, a 5 x 5 predicted-field patch, and the latest source-support mask. By default, outputs are written to:

```text
${SATPV_DATA_ROOT}/stage4/runs/05l_per_station_temporal_stage3_field_patch_seed2026/
```

The temporal, nearest-neighbor, STGCN, AGCRN, and other manuscript baselines are exposed through the launchers under `stage_4/run_stage4_*.sh`.

## Unified station evaluation

The manuscript station table is computed on the exact intersection of `(station, issue time, horizon)` keys shared by all methods:

```bash
python experiments/unified_station_evaluation/evaluate_common_intersection.py \
  --old-data-root "${SATPV_DATA_ROOT}" \
  --output-dir "${SATPV_DATA_ROOT}/evaluation/common_intersection"
```

The evaluator writes pooled metrics, horizon-wise metrics, station-wise metrics, an audit report, and paired-bootstrap comparisons. Use `--method-path key=/path/to/eval_detail.csv` when a result file is stored outside the conventional directory.

## Supplementary experiments

### Spatial representation consistency

```bash
PV_ROOT="${PV_ROOT}" \
OLD_DATA_ROOT="${SATPV_DATA_ROOT}" \
bash paper_support/run_spatial_representation_suite.sh
```

### Matched graph-versus-field representation comparison

```bash
OLD_DATA_ROOT="${SATPV_DATA_ROOT}" \
bash paper_support/matched_representation/run_audit.sh

OLD_DATA_ROOT="${SATPV_DATA_ROOT}" \
bash paper_support/matched_representation/run_phase2.sh

OLD_DATA_ROOT="${SATPV_DATA_ROOT}" \
bash paper_support/matched_representation/run_full_experiment.sh all
```

Run the audit first; the full experiment assumes that Phase 1 and Phase 2 both pass.

## Determinism and evaluation protocol

- The paper experiments use seed `2026` unless an individual script explicitly documents another seed.
- Split boundaries are `2023-01-13 23:59:59` for training and `2023-03-08 23:59:59` for validation; later samples are test samples.
- Stage 3 issue origins are hourly during the configured daytime range, while PV samples remain 15-min resolution.
- Station-level PV is normalized by installed capacity. Regional-field values additionally use a source-station training q95 scale and clipping at 1.5.
- Field losses and field metrics use the source-support mask. Station metrics are reported after inverse standardization in installed-capacity-normalized units.
- The unified evaluator verifies that ground-truth values agree across methods before calculating the common intersection.

## Outputs intentionally excluded from Git

The `.gitignore` excludes:

- raw and processed data;
- station CSVs and regional caches;
- NumPy/NetCDF/HDF5/raster artifacts;
- checkpoints (`.pt`, `.pth`, `.ckpt`);
- `runs/`, `outputs/`, and Python caches.

Before publishing, check the staged files:

```bash
git status --short
git ls-files | grep -E '\.(pt|pth|ckpt|npy|npz|nc|csv|xlsx|h5|hdf5)$' && \
  echo "Unexpected data artifact found" && exit 1 || true
```

## Troubleshooting

### `externally-managed-environment`

Create a virtual environment rather than installing into the system interpreter. If `ensurepip` is unavailable, install `python3-venv` first. The provided `setup_env_tuna.sh` follows this approach.

### Stage 3 cannot find an active future-SWR cache

Run:

```bash
bash stage_3/run_stage3_build_future_swr_cache.sh
```

The cache manifest and checkpoint provenance must agree.

### Stage 4 cannot find the field checkpoint

Check that `SATPV_DATA_ROOT` points to the same artifact root used for Stage 3 and that the final checkpoint uses the expected relative path shown above.

### CUDA/cuDNN LSTM instability

The final Stage 4 configuration disables the cuDNN RNN path by default to match the stable experiment execution. This is intentional and can be overridden only after validating numerical equivalence.

### Figure scripts need geographic data

Natural Earth shapefiles and downloaded map tiles are not bundled. Set the relevant command-line path or place the legally obtained files under the location described by the individual figure script. Core training does not require geographic plotting packages or map files.

## Citation and license

The source code is released under the [MIT License](LICENSE). The license covers the repository code only; it does not grant rights to the operational PV data, satellite products, NWP inputs, model weights, map assets, or other third-party material.

The final article citation and `CITATION.cff` will be added after publication.

