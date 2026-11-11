#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${RAP_ENC_KEY:?Set RAP_ENC_KEY externally}"
: "${RAP_XOR_KEY:?Set RAP_XOR_KEY externally}"
: "${RAP_STC_KEY:?Set RAP_STC_KEY externally}"
export RAP_BACKEND="${RAP_BACKEND:-CPU}"
export RAP_REPORT_DIR="${RAP_REPORT_DIR:-results}"
CPU_SLOTS="${CPU_SLOTS:-$(nproc)}"
MANIFEST="${MANIFEST:-$RAP_REPORT_DIR/batch_manifests/batch_$(date +%Y%m%d_%H%M%S)_${PAIR_REMAINDER:-0}_$$.csv}"
args=(batch-embed --pngDir "${INPUT_DIR:-input}" --msgDir "${MSG_DIR:-msg}" --outDir "${OUTPUT_DIR:-output}"
  --encKeyEnv RAP_ENC_KEY --xorKeyEnv RAP_XOR_KEY --stcKeyEnv RAP_STC_KEY
  --geometryMode "${GEOMETRIES:-ALL_GEOMETRY}" --stcHeight "${STC_HEIGHT:-10}"
  --workers "${WORKERS:-auto}" --threadMode "${THREAD_MODE:-virtual}"
  --carrierParallelism "${CARRIER_PARALLELISM:-$CPU_SLOTS}" --cpuSlots "$CPU_SLOTS"
  --continueOnError "${CONTINUE_ON_ERROR:-true}" --recursive "${RECURSIVE:-true}" --resume "${RESUME:-false}"
  --startIndex "${START_INDEX:-1}" --pairModulo "${PAIR_MODULO:-1}" --pairRemainder "${PAIR_REMAINDER:-0}"
  --shardUnit "${SHARD_UNIT:-IMAGE}" --manifest "$MANIFEST" --messageExtension "${MESSAGE_EXTENSION:-.msg}"
  --auditEnabled "${AUDIT_ENABLED:-true}" --auditMode "${AUDIT_MODE:-RESULT_ONLY}" --auditProfile "${AUDIT_PROFILE:-THESIS}"
  --deleteJpegsAfterEvaluation "${DELETE_JPEGS_AFTER_EVALUATION:-false}")
[[ -z "${ONLY_MESSAGE_NAME:-}" ]] || args+=(--onlyMessageName "$ONLY_MESSAGE_NAME")
[[ $# -eq 0 ]] || { echo 'Configure batch.sh through environment variables; use run.sh for CLI options' >&2; exit 2; }
if [[ -n "${CPU_AFFINITY:-}" ]]; then
  command -v taskset >/dev/null || { echo 'CPU_AFFINITY requires taskset' >&2; exit 2; }
  args+=(--cpuAffinity "$CPU_AFFINITY")
  exec taskset -c "$CPU_AFFINITY" "$ROOT/run.sh" "${args[@]}"
fi
exec "$ROOT/run.sh" "${args[@]}"
