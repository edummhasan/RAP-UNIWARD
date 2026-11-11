#!/usr/bin/env bash
# One process per GPU, with deterministic image sharding and optional CPU retry.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BACKEND="${RAP_BACKEND:-CUDA}"; BACKEND="${BACKEND^^}"
[[ "$BACKEND" == CUDA || "$BACKEND" == HIP ]] || { echo 'RAP_BACKEND must be CUDA or HIP' >&2; exit 2; }
GPU_LIST="${GPU_IDS-0}"
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPU_IDS must be a comma-separated list of nonnegative integers' >&2; exit 2; }
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ ${#GPUS[@]} -gt 0 ]] || { echo 'GPU_IDS must not be empty' >&2; exit 2; }
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z "${SEEN[$gpu]:-}" ]] || { echo 'GPU_IDS must contain distinct nonnegative integer device IDs' >&2; exit 2; }
  SEEN[$gpu]=1
done
[[ "${DELETE_JPEGS_AFTER_EVALUATION:-false}" == false ]] || { echo 'Multi-GPU resume requires DELETE_JPEGS_AFTER_EVALUATION=false' >&2; exit 2; }
[[ "${START_INDEX:-1}" == 1 ]] || { echo 'Multi-GPU full-grid merge requires START_INDEX=1' >&2; exit 2; }
: "${RAP_ENC_KEY:?Set RAP_ENC_KEY externally}"; : "${RAP_XOR_KEY:?Set RAP_XOR_KEY externally}"; : "${RAP_STC_KEY:?Set RAP_STC_KEY externally}"
FAILOVER="${RAP_FAILOVER_POLICY:-NONE}"
case "$FAILOVER" in NONE|CPU|GPU_THEN_CPU) ;; *) echo 'RAP_FAILOVER_POLICY must be NONE, CPU or GPU_THEN_CPU' >&2; exit 2;; esac
N="${#GPUS[@]}"; HOST_CPUS="$(nproc)"
SLOTS="${SHARD_CPU_SLOTS:-$(( HOST_CPUS / N > 0 ? HOST_CPUS / N : 1 ))}"
WAVE_OPTS="${JAVA_OPTS:--Djava.awt.headless=true -XX:MaxRAMPercentage=$(( 70 / N > 1 ? 70 / N : 1 ))}"
RUN_ROOT="${RUN_ROOT:-results/multi_gpu_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
[[ ! -e "$RUN_ROOT/invocation.lock" ]] || { echo 'Use a new RUN_ROOT for each multi-GPU invocation' >&2; exit 2; }
(set -C; : > "$RUN_ROOT/invocation.lock")
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/output}"
pids=(); manifests=(); LAUNCHED_PID=''
declare -A running=()
stop_children() {
  local code="$1"
  trap '' INT TERM
  for pid in "${!running[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  exit "$code"
}
trap 'stop_children 130' INT
trap 'stop_children 143' TERM
launch() {
  local shard="$1" gpu="$2" backend="$3" resume="$4" phase="$5"
  local report="$RUN_ROOT/shard_$shard/$phase"
  mkdir -p "$report"
  local -a environment=(RAP_BACKEND="$backend" RAP_PHYSICAL_GPU_ID="$gpu" RAP_CUDA_DEVICE=0
    RAP_ACCELERATOR_FALLBACK=false RAP_REPORT_DIR="$report" RAP_PREPARED_CACHE_DIR="$report/cache"
    PAIR_MODULO="$N" PAIR_REMAINDER="$shard" SHARD_UNIT=IMAGE RESUME="$resume" CONTINUE_ON_ERROR=true
    MANIFEST="$report/manifest.csv" CPU_SLOTS="$SLOTS" CARRIER_PARALLELISM="$SLOTS" JAVA_OPTS="$WAVE_OPTS")
  if [[ "$backend" == CUDA ]]; then environment+=(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"); fi
  if [[ "$backend" == HIP ]]; then environment+=(ROCR_VISIBLE_DEVICES="$gpu"); fi
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES "${environment[@]}" \
    "$ROOT/scripts/batch.sh" > "$report/console.log" 2>&1 &
  LAUNCHED_PID=$!
  pids+=("$LAUNCHED_PID")
  running[$LAUNCHED_PID]=1
  manifests[$shard]="$report/manifest.csv"
}
for ((r=0; r<N; r++)); do
  launch "$r" "${GPUS[$r]}" "$BACKEND" false initial
  echo "Started shard=$r backend=$BACKEND device=${GPUS[$r]} pid=$LAUNCHED_PID"
done
failed=()
for ((r=0; r<N; r++)); do
  if ! wait "${pids[$r]}"; then failed+=("$r"); fi
  unset "running[${pids[$r]}]"
done
for r in "${failed[@]}"; do
  [[ "$FAILOVER" != NONE ]] || { echo "Shard $r failed; see $RUN_ROOT" >&2; exit 3; }
  if [[ "$FAILOVER" == GPU_THEN_CPU && "$N" -gt 1 ]]; then
    launch "$r" "${GPUS[$(((r+1)%N))]}" "$BACKEND" true retry_gpu
    if wait "$LAUNCHED_PID"; then
      unset "running[$LAUNCHED_PID]"
      continue
    fi
    unset "running[$LAUNCHED_PID]"
  fi
  launch "$r" CPU CPU true retry_cpu
  wait "$LAUNCHED_PID" || { echo "CPU retry failed for shard $r" >&2; exit 4; }
  unset "running[$LAUNCHED_PID]"
done
python3 "$ROOT/tools/merge_manifests.py" "$RUN_ROOT/manifest_merged.csv" "${manifests[@]}"
echo "Completed: $RUN_ROOT/manifest_merged.csv"
