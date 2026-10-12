# Paper-to-code map

This file identifies the authoritative entry points for the manuscript. Files ending in `_impl.py` are implementation modules imported by wrappers; they are not separate experimental protocols.

## Method pipeline

| Manuscript component | Authoritative code |
|---|---|
| H8 preprocessing and sample alignment | `stage_1/run_stage1_build_all.sh` |
| Ratio3DCNN irradiance model | `stage_2/train_3dcnn_ratio_big_patch_5x5.py` |
| Residual ConvLSTM correction | `stage_2/train_residual_convlstm_v2_no_center_weight.py` |
| Source/target split and PV-field cache | `stage_3/02_select_targets_build_pvnorm_cache.py` |
| Hourly irradiance-anchor cache | `stage_3/03_build_center_h8_swr_anchor_cache.py` |
| Fixed temporal anchor weights | `stage_3/stage3_motion_fusion.py` |
| Final field forecaster | `stage_3/ablation_motion_branch/train_motion_branch.py` with `last_only`, `warp_direct_gate`, `full_swr_future` |
| Final target-station forecaster | `stage_4/05l_train_per_station_temporal_with_stage3_field_patch.py` |

## Main experimental comparisons

| Experiment | Code |
|---|---|
| Stage 3 field baselines | `stage_3/run_stage3_train_baseline_models.sh`, `stage_3/run_stage3_eval_baseline_models.sh` |
| Stage 3 component ablations | `stage_3/ablation_*` |
| Stage 4 temporal baselines | `stage_4/19_train_stage4_nwp_backbone_baselines.py` |
| STGCN | `stage_4/20_train_stage4_stgcn_forecast_weather_baseline.py` |
| AGCRN | `stage_4/23_train_stage4_graph_nwp_baselines.py` |
| 3-nearest-source guidance | `stage_4/21_train_stage4_nearest3_station_pv_baseline.py` |
| Historical/future field guidance | `stage_4/05h_train_per_station_temporal_with_historical_field_patch.py`, `stage_4/05l_train_per_station_temporal_with_stage3_field_patch.py` |
| Irradiance-guidance control | `stage_4/06l_train_per_station_temporal_with_stage3_swr_patch.py` |
| Common-intersection metrics and paired bootstrap | `experiments/unified_station_evaluation/evaluate_common_intersection.py` |
| Operating-condition analysis | `stage_4/25_eval_stage4_ramp_condition_analysis.py` |
| Source-fleet density | `stage_3/source_fleet_density`, `stage_4/source_fleet_density` |
| Matched graph-versus-field comparison | `paper_support/matched_representation` |
| Spatial representation consistency | `paper_support/run_spatial_representation_consistency.py` |

## Important implementation details

- Regional fields contain only source-station PV measurements. The full coordinate table may define geographical bounds, but target PV values are excluded before splatting.
- Regional field values are capacity-normalized, divided by a station training-period q95 scale, and clipped to 1.5.
- Station-level targets use installed-capacity normalization and station-specific train standardization for optimization.
- The Stage 3 fixed hourly-to-15-min alignment matrix is 8 x 4 and acts on the latest observed plus three future irradiance anchors.
- The final Stage 3 branch predicts horizon-specific displacement, direct field, and pixel-wise fusion gate outputs.
- The Stage 4 patch tensor is `8 x 2 x 5 x 5`: standardized predicted-field values plus the repeated latest binary support mask.
- Weighted station SmoothL1 divides by the sum of horizon weights. The large-change condition is evaluated in installed-capacity-normalized PV space.

## Naming

The public paper-level method name is **SatPV-Field**. Some source filenames and historical result fields retain earlier internal identifiers for checkpoint compatibility. Renaming those symbols would break artifact loading and is intentionally avoided in this release.

