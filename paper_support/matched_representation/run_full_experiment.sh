#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_ROOT="${MATCHED_OUTPUT_ROOT:-${PROJECT_ROOT}/paper_support/matched_representation/outputs/full}"
AUDIT_DIR="${AUDIT_DIR:-${PROJECT_ROOT}/paper_support/matched_representation/outputs/audit}"
SOURCE_DIR="${OUTPUT_ROOT}/source_graph"
TARGET_DIR="${OUTPUT_ROOT}/target_graph"
SUMMARY_DIR="${OUTPUT_ROOT}/summary"
STAGE="${1:-all}"

COMMON_ARGS=(--project-root "${PROJECT_ROOT}" --audit-dir "${AUDIT_DIR}")
if [[ -n "${OLD_DATA_ROOT:-}" ]]; then COMMON_ARGS+=(--old-data-root "${OLD_DATA_ROOT}"); fi
if [[ -n "${PV_ROOT:-}" ]]; then COMMON_ARGS+=(--pv-root "${PV_ROOT}"); fi
if [[ -n "${STAGE3_DATA_DIR:-}" ]]; then COMMON_ARGS+=(--stage3-data-dir "${STAGE3_DATA_DIR}"); fi
if [[ -n "${STAGE3_MODEL_DIR:-}" ]]; then COMMON_ARGS+=(--stage3-model-dir "${STAGE3_MODEL_DIR}"); fi
if [[ -n "${TRAIN_PV_DIR:-}" ]]; then COMMON_ARGS+=(--train-pv-dir "${TRAIN_PV_DIR}"); fi
if [[ -n "${VAL_PV_DIR:-}" ]]; then COMMON_ARGS+=(--val-pv-dir "${VAL_PV_DIR}"); fi

run_source() {
  SOURCE_ARGS=(
    "${COMMON_ARGS[@]}"
    --output-dir "${SOURCE_DIR}"
    --device "${DEVICE:-cuda}"
    --seed "${SEED:-2026}"
    --batch-size "${SOURCE_BATCH_SIZE:-8}"
    --epochs "${SOURCE_EPOCHS:-80}"
    --patience "${SOURCE_PATIENCE:-10}"
    --hidden-dim "${SOURCE_HIDDEN_DIM:-96}"
    --num-workers "${SOURCE_NUM_WORKERS:-0}"
  )
  if [[ "${SOURCE_CACHE_ONLY:-0}" == "1" ]]; then SOURCE_ARGS+=(--cache-only); fi
  "${PYTHON}" "${PROJECT_ROOT}/paper_support/matched_representation/train_source_graph_full.py" "${SOURCE_ARGS[@]}"
}

run_target() {
  TARGET_ARGS=(
    "${COMMON_ARGS[@]}"
    --source-dir "${SOURCE_DIR}"
    --output-dir "${TARGET_DIR}"
    --device "${DEVICE:-cuda}"
    --seed "${SEED:-2026}"
    --batch-size "${TARGET_BATCH_SIZE:-128}"
    --epochs "${TARGET_EPOCHS:-60}"
    --patience "${TARGET_PATIENCE:-10}"
  )
  if [[ "${RESUME_TARGET:-0}" == "1" ]]; then TARGET_ARGS+=(--resume); fi
  if [[ "${TARGET_LIMIT:-0}" != "0" ]]; then TARGET_ARGS+=(--target-limit "${TARGET_LIMIT}"); fi
  "${PYTHON}" "${PROJECT_ROOT}/paper_support/matched_representation/train_target_graph_full.py" "${TARGET_ARGS[@]}"
}

run_summary() {
  SUMMARY_ARGS=(
    --graph-detail "${TARGET_DIR}/target_graph_eval_detail.csv"
    --source-metrics "${SOURCE_DIR}/source_graph_metrics.csv"
    --target-run-meta "${TARGET_DIR}/run_meta.json"
    --output-dir "${SUMMARY_DIR}"
    --seed "${SEED:-2026}"
  )
  if [[ -n "${FIELD_HIST_DETAIL:-}" ]]; then SUMMARY_ARGS+=(--historical-field-detail "${FIELD_HIST_DETAIL}"); fi
  if [[ -n "${FIELD_PRED_DETAIL:-}" ]]; then SUMMARY_ARGS+=(--predicted-field-detail "${FIELD_PRED_DETAIL}"); fi
  if [[ -n "${EXTERNAL_RESULTS:-}" ]]; then SUMMARY_ARGS+=(--external-results "${EXTERNAL_RESULTS}"); fi
  "${PYTHON}" "${PROJECT_ROOT}/paper_support/matched_representation/summarize_matched_representation.py" "${SUMMARY_ARGS[@]}"
}

run_table() {
  TABLE_ARGS=(
    --project-root "${PROJECT_ROOT}"
    --graph-detail "${TARGET_DIR}/target_graph_eval_detail.csv"
    --output-dir "${OUTPUT_ROOT}/table_iv"
  )
  if [[ -n "${OLD_DATA_ROOT:-}" ]]; then TABLE_ARGS+=(--old-data-root "${OLD_DATA_ROOT}"); fi
  if [[ -n "${FIELD_HIST_DETAIL:-}" ]]; then TABLE_ARGS+=(--historical-field-detail "${FIELD_HIST_DETAIL}"); fi
  if [[ -n "${FIELD_PRED_DETAIL:-}" ]]; then TABLE_ARGS+=(--predicted-field-detail "${FIELD_PRED_DETAIL}"); fi
  "${PYTHON}" "${PROJECT_ROOT}/paper_support/matched_representation/export_table_iv_values.py" "${TABLE_ARGS[@]}"
}

case "${STAGE}" in
  source) run_source ;;
  target) run_target ;;
  summary) run_summary ;;
  table) run_table ;;
  all)
    run_source
    run_target
    run_summary
    run_table
    ;;
  *)
    echo "Usage: bash paper_support/matched_representation/run_full_experiment.sh [source|target|summary|table|all]" >&2
    exit 2
    ;;
esac
