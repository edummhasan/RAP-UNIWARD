# Runtime reference

This contract describes RAP-UNIWARD 1.0.0.1. The source-defined terminal CSV column
order is available in [result-schema.json](result-schema.json).

## Entry points and precedence

| Entry point | Function |
| --- | --- |
| `build.sh` | Build Java, tests, JAR and selected native libraries |
| `build.sh --java-only` | Compile Java/tests/JAR without building native libraries; diagnostic only |
| `run.sh COMMAND ...` | Invoke the CLI from the repository root |
| `scripts/batch.sh` | Configure a batch through environment variables |
| `scripts/background.sh start\|status\|tail\|stop` | Manage one detached batch with PID identity checks |
| `scripts/multi-gpu.sh` | Process-isolated CUDA or HIP image sharding |
| `test.sh` | Repository checks, Python workflow checks, native tests and Java test programs |
| `tools/check_thesis.py` | Read-only LaTeX reference and summary-evidence reconciliation |

CLI options use **separate name/value arguments**, e.g. `--resume true`.
`--resume=true`, duplicate options and unknown options are rejected. The wrapper
uses environment variables to construct the corresponding CLI arguments.
Where an option has an environment fallback, the CLI value takes precedence.
Java system properties take precedence over corresponding engine environment
settings. `RAP_BACKEND`/`rap.backend`, when explicitly set, overrides the SI and
STC per-stage backend selectors. The batch wrapper sets `RAP_BACKEND=CPU` unless
configured otherwise. Direct Java/`run.sh` execution defaults to engine `AUTO`.

## Build settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `RAP_BUILD_BACKEND` | `CPU` | `CPU`, `CUDA`, `HIP`, `ALL`, or `AUTO`; `ROCM` aliases `HIP` |
| `RAP_JAVA_RELEASE` | `22` | Java target, at least 21; compiler must support it |
| `JAVA_HOME`, `PATH` | Host configuration | JDK and compiler/JNI include location |
| `JAVA_BIN` | `java` | Java launcher for execution and module-tool fallback |
| `JPEG_INCLUDE_DIR` | `/usr/include` | Directory containing `jpeglib.h` |
| `JPEG_LIB_DIR` | Linker default | Optional libjpeg library directory |
| `CUDA_ROOT`, `NVCC` | `/usr/local/cuda`, `$CUDA_ROOT/bin/nvcc` | CUDA compiler/toolkit |
| `RAP_CUDA_ARCH` | sm_80/86/89/90 and compute_90 PTX | Explicit `sm_XX` selects one architecture; toolkit must support the requested targets |
| `ROCM_ROOT`, `HIPCC` | `/opt/rocm`, `$ROCM_ROOT/bin/hipcc` | ROCm compiler/toolkit |
| `RAP_HIP_ARCH` | Required for HIP | Actual device target; no hardware model is assumed |

A full build requires `jni.h` and libjpeg development files before compiling.
A Java-only check can use a matching modular compiler/runtime if its `ct.sym`
release catalog is absent. Such a check does not establish native integration.

## Command options

`embed`, `batch-embed` and `extract` require three keys. Use the environment-option
trio `--encKeyEnv RAP_ENC_KEY --xorKeyEnv RAP_XOR_KEY --stcKeyEnv RAP_STC_KEY`.
The direct-value trio `--encKey`, `--xorKey`, `--stcKey` is also accepted, but exposes
secrets in process arguments and shell history. Do not mix the two forms.

| Command | Required paths | Additional options |
| --- | --- | --- |
| `embed` | `--png`, `--msgFile`, `--out` | Geometry, STC height, fixed QF/policy checks, audit settings |
| `batch-embed` | `--pngDir`, `--msgDir`, `--outDir` | All batch, geometry, concurrency and audit settings below |
| `extract` | `--png`, `--stego`, `--outFile` | Geometry, fixed QF interval, `--stcHeight auto\|10\|12` |
| `summarize` | Defaults below | `--runs`, `--positions`, `--channels`, `--channelPositions`, `--attempts`, `--outDir` |
| `detector-summary` | `--csv` | `--labelColumn` (label), `--scoreColumn` (score), `--out` |
| `--help`, `--version` | None | Print usage or application identity |

| Shared option | Default / allowed values | Contract |
| --- | --- | --- |
| `--geometryMode` | `ALL_GEOMETRY`; single or comma-separated subset | `444,422,420,440,411,441,410`; colon labels also accepted |
| `--minQf`, `--maxQf` | `75`, `100` | Fixed conformance range; other values are rejected |
| `--selectionPolicy` | `capacity` | Capacity prefilter followed by full verification; no extra quality/security threshold |
| `--stcHeight` | Embed: 10; extract: auto | Embed accepts 10 or 12; auto extraction tries 10 then 12 |
| `--auditEnabled` | `true` | `true`/`false`; core verification remains mandatory |
| `--auditMode` | `RESULT_ONLY` | `RESULT_ONLY` or `QF_FORENSIC` |
| `--auditProfile` | `THESIS` | In forensic mode: `THESIS`, `CHANGES`, `FULL_AUDIT` |

The optional environment fallbacks are `RAP_STC_HEIGHT`, `RAP_AUDIT_ENABLED`,
`RAP_AUDIT_MODE` and `RAP_AUDIT_PROFILE`. The batch wrapper uses `STC_HEIGHT` and
`AUDIT_*` to set explicit CLI values; those wrapper values therefore override the
corresponding engine fallbacks.

## Batch wrapper and concurrency

| Environment variable | CLI option / role | Wrapper default |
| --- | --- | --- |
| `INPUT_DIR`, `MSG_DIR`, `OUTPUT_DIR` | `--pngDir`, `--msgDir`, `--outDir` | `input`, `msg`, `output` |
| `GEOMETRIES` | `--geometryMode` | `ALL_GEOMETRY` |
| `STC_HEIGHT` | `--stcHeight` | `10` |
| `THREAD_MODE` | `--threadMode` | `virtual`; `platform` also supported |
| `WORKERS` | `--workers` | `auto` in virtual mode; positive integer for platform mode |
| `CPU_SLOTS` | `--cpuSlots` | `nproc` |
| `CARRIER_PARALLELISM` | `--carrierParallelism` | CPU slots |
| `CPU_AFFINITY` | `taskset -c` and `--cpuAffinity` | Unset |
| `RECURSIVE` | `--recursive` | `true` |
| `MESSAGE_EXTENSION` | `--messageExtension` | `.msg`; must start with a dot |
| `ONLY_MESSAGE_NAME` | `--onlyMessageName` | Unset; must resolve to exactly one file if supplied |
| `RESUME` | `--resume` | `false` |
| `CONTINUE_ON_ERROR` | `--continueOnError` | `true` |
| `START_INDEX` | `--startIndex` | `1`, one-based terminal-condition index |
| `PAIR_MODULO`, `PAIR_REMAINDER` | Shard ownership | `1`, `0`; remainder in 0..modulo−1 |
| `SHARD_UNIT` | `--shardUnit` | `IMAGE`; direct CLI defaults to `PAIR` |
| `MANIFEST` | `--manifest` | Unique CSV under `$RAP_REPORT_DIR/batch_manifests/` |
| `AUDIT_ENABLED`, `AUDIT_MODE`, `AUDIT_PROFILE` | Explicit audit CLI values | `true`, `RESULT_ONLY`, `THESIS` |
| `DELETE_JPEGS_AFTER_EVALUATION` | `--deleteJpegsAfterEvaluation` | `false`; incompatible with resume |
| `RAP_REPORT_DIR` | Ledgers/audit root | `results`; property override `rap.report.dir` |
| `RAP_JAR` | JAR path | `build/rap-uniward-1.0.0.1.jar` |
| `JAVA_OPTS` | JVM arguments | `-Djava.awt.headless=true` |

Direct CLI batch execution defaults to `platform` mode with one worker, unlike the
virtual-thread wrapper. Virtual `WORKERS=auto` uses a bounded in-flight window;
it does not schedule the complete dataset at once. CPU slots limit simultaneous
whole-condition kernels. `CONTINUE_ON_ERROR=false` deliberately uses sequential
fail-fast execution. Concurrency/resource values are validated; platform CPU slots
cannot exceed its worker count.

`--cpuAffinity` alone records a declaration; it does not set operating-system
CPU affinity. The wrapper's `CPU_AFFINITY` does apply `taskset` and records it.
`JAVA_OPTS` is whitespace-separated without shell evaluation; use `JAVA_TOOL_OPTIONS`
for JVM-supported quoting when needed. Never place keys in JVM options.

## Engine and accelerator environment

| Setting | Default | Purpose |
| --- | --- | --- |
| `RAP_BACKEND` | Engine AUTO; wrapper CPU | Process-wide CPU/CUDA/HIP/AUTO policy |
| `RAP_ACCELERATOR_FALLBACK` | false | Permit CPU fallback in explicit CUDA/HIP mode |
| `RAP_BALANCE_PROFILE` | B | U unrestricted; B permits absolute net zero-flow ≤4; S requires 0 |
| `RAP_TRUE_PARALLEL_CACHE` | true | Reuse immutable source/geometry/QF preparation |
| `RAP_PREPARED_CACHE_STATES` | 32 | Prepared state cache bound, 1–4096 |
| `RAP_SOURCE_CACHE_IMAGES` | 2 | Source image cache bound, 1–128 |
| `RAP_QF_PREFETCH_DEPTH` | 4 | Speculative QF preparation, 0–26; 0 disables |
| `RAP_QF_PREFETCH_WORKERS` | 4 | Prefetch pool, clamped to 1–16 |
| `RAP_PREPARED_CACHE_DIR` | `build/true_parallel_prepared_cache` | Temporary preparation directory |
| `RAP_KEEP_PREPARED_CACHE` | false | Retain preparation directory where supported |
| `RAP_SI_BACKEND` | AUTO | Per-stage CPU/HIP/CUDA/AUTO/VERIFY/VERIFY_HIP/VERIFY_CUDA; overridden by explicit process backend |
| `RAP_STC_BACKEND` | AUTO | Per-stage CPU/CUDA/AUTO; overridden by explicit process backend |
| `RAP_TRUE_GPU_STRICT` | false | Per-stage strictness when no process backend is explicitly configured |
| `RAP_SI_CUDA_MIN_CANDIDATES` | 8192 | CUDA cost-batch threshold |
| `RAP_SI_HIP_MIN_CANDIDATES` | 8192 | HIP cost-batch threshold |
| `RAP_STC_CUDA_MIN_CARRIERS` | 4096 | Small-call CPU routing threshold even with an available CUDA codec; nonnegative |
| `RAP_GPU_SLOTS` | 1 | SI accelerator admission slots; not a GPU count |
| `RAP_CUDA_INFLIGHT` | 16 | SI CUDA in-flight limit; subject to the global guard |
| `RAP_CUDA_GLOBAL_INFLIGHT` | 1 | Process-wide concurrent CUDA JNI limit, 1–64 |
| `RAP_CUDA_STREAMS` | 8 | Native stream/workspace pool size, 1–64 |
| `RAP_CUDA_STC_STREAMS` | CUDA stream setting | STC-specific native stream pool, 1–64 |
| `RAP_CUDA_DEVICE` | 0 | Logical CUDA device within process visibility |
| `RAP_PHYSICAL_GPU_ID` | CPU_OR_UNDECLARED | Reporting label; does not select a device |
| `RAP_GPU_VERIFY_VERBOSE` | false | Verbose exact SI parity verification |
| `RAP_GPU_PREPROCESS`, `RAP_GPU_DCT` | true | Enable eligible CUDA preprocessing/DCT stages |
| `RAP_GPU_CANONICAL_LUMA`, `RAP_GPU_CANDIDATE_SORT` | true | Enable eligible CUDA reference-luma and sorting stages |
| `RAP_GPU_REDUCTIONS`, `RAP_GPU_CRYPTO` | true | Enable eligible CUDA statistical reduction/HMAC stages |
| `RAP_GPU_BALANCE_MEDIAN`, `RAP_GPU_BALANCE_PLAN` | true | Enable eligible CUDA balance stages |

Stage flags do not bypass backend permission, availability checks or exactness
requirements. A stream count above one does not override the default global
single-call CUDA guard. Do not report parallel GPU speedups from configuration
alone. The example CPU environment sets smaller caches and disables prefetch.

## Multi-GPU and background control

`GPU_IDS` is a comma-separated list of distinct numeric physical IDs, default `0`.
`RAP_BACKEND` must be CUDA or HIP. `SHARD_CPU_SLOTS` defaults to the host CPU count
split over workers. Each process receives separate reports/cache/manifest paths,
plus shared output ownership enforced by deterministic image sharding.

`RAP_FAILOVER_POLICY=NONE` is the default; `CPU` and `GPU_THEN_CPU` enable retries.
Retries are new processes with authenticated resume, not live device migration.
Use a fresh `RUN_ROOT` per invocation. Completed shard manifests are checked for
duplicates and full-grid coverage before atomic publication of the merged CSV.
`RAP_ALLOW_INCOMPLETE_MERGE=true` is a separate merger-only option for a deliberately
partial set. Multi-GPU execution requires START_INDEX=1 and retained JPEGs.

`CONTROL_DIR` defaults to `logs/background`. `LINES` controls tail output (100 by
default). The background controller serializes state changes and checks the process
start identity before sending a signal; it will not overwrite an active job's state.

## Input and output contract

Inputs are actual PNG files, verified by their signature. They are decoded with
ImageIO and normalized to `TYPE_INT_RGB`; alpha-bearing inputs are composited on
white by Java2D. For reproducible experiments, use already-normalized 8-bit RGB
PNGs and matching Java/ImageIO behavior at both endpoints. The implementation does
not enforce original PNG bit depth as a separate input restriction.

Message files contain arbitrary bytes, including empty content. Raw messages can
contain at most 65,526 bytes in the compact frame. Longer messages only fit if the
complete deterministic Huffman representation fits the 65,535-byte prepared
ciphertext bound. Compression is selected only if its complete serialization is
strictly smaller than the raw message.

Batch discovery is recursive by default and sorts relative paths lexically.
Output names encode the lengths and sanitized relative identity strings. Example:
`input/000001.png` and `msg/001.msg`, geometry 420, produce
`output/i6_000001__m3_001__g420_stego.jpg`. Colliding sanitized output names are
rejected before embedding. Output and input roots must not overlap.

| Artifact | When written | Meaning |
| --- | --- | --- |
| `output/*_stego.jpg` | Accepted output | Verified JPEG coefficient state and authenticated message |
| `output/*.jpg.verified` | Accepted output | Hashes, profile, geometry and release identity used for resume |
| `results/audit_final_qf.csv` | RESULT_ONLY | One terminal record per executed condition; repeats append |
| `results/batch_manifests/*.csv` | Batch | Operational terminal statuses, pair indices and output paths |
| `results/evaluation_runs.csv` | QF_FORENSIC success | Run-level evaluation table |
| `results/evaluation_attempts.csv` | QF_FORENSIC | Attempt records |
| `results/evaluation_json/<run_id>.json` | QF_FORENSIC success | Structured successful-run metrics |
| `results/audit/<run_id>/` | QF_FORENSIC | `run.json`, `qf_trials.csv`, `balance_attempts.csv`, candidate/JPEG summaries, verification |
| `results/audit/<run_id>/candidate_changes.csv` | CHANGES or FULL_AUDIT | Detailed modified-candidate records |
| `results/audit/<run_id>/candidates_full.csv.gz` | FULL_AUDIT | Complete candidate evidence |
| `results/evaluation_zz_positions.csv`, `evaluation_channels.csv`, `evaluation_channel_zz_positions.csv` | QF_FORENSIC + FULL_AUDIT | Expanded coordinate/component tables |
| `results/qf_band_reports/` | Forensic reporting | QF band/capacity diagnostics |
| `build/test-results/` | Tests | Software check results; not experiment observations |

In RESULT_ONLY, `AUDIT_PROFILE` does not expand the ledger into per-coefficient
files. Audit-disabled runs still verify outputs and write batch operational state.
Reference JPEGs are temporary preparation artifacts and are not a preserved
cover/stego dataset by default. `DELETE_JPEGS_AFTER_EVALUATION=true` is an explicit
storage choice that removes accepted JPEGs/sidecars after committed processing.

CSV identifiers remain strings. Empty cells mean unavailable, not zero. A failure
row can have a terminal tested QF without any accepted selected QF. Operational
manifest statuses (`COMPLETED`, `VERIFIED_EXISTING`, `NO_OUTPUT_CHECK_ATTEMPT_LEDGER`,
etc.) are distinct from the final ledger's scientific status and reason fields.
A batch may exit successfully with valid no-output conditions; unexpected failures
produce a nonzero exit. Review the manifest/ledger, not only the process exit code.

Forensic summary example:

```bash
AUDIT_MODE=QF_FORENSIC AUDIT_PROFILE=FULL_AUDIT ./scripts/batch.sh
./run.sh summarize
```

The summary command reads the configured results directory and writes
`experiment_summary/` below it. It does not deduplicate repeated experimental
conditions or reproduce clustered bootstrap confidence intervals. Detector-summary
requires externally obtained labelled scores; it does not train or run a detector.

## Immutable protocol bytes

The production AAD domain is `AHP-PAYLOAD-AAD-V14.2.0-GEOM-V1`; its 31 bytes plus
QF (1), geometry signature (4), ciphertext length (4) and XOR-key commitment (32)
produce 72 authenticated, non-embedded bytes. The ordering and tie-break domains
are `ahp-order-v1` and `ahp-direction-v1`. These exact identifiers are required by
the uploaded methodology and are not application release-history labels.
A non-geometry payload helper API also retains its fixed AAD domain; the production
CLI uses the geometry-bound path. Changing these bytes changes the protocol.

The embedded frame is uint16 ciphertext length, 12-byte nonce, 12-byte tag, then
ciphertext. The 26-byte header occupies 208 bits and 240 carriers including the
32-carrier reserve. The nine-byte preprocessing header brings raw-message framing
overhead to 35 bytes. Body width cap is 256 at heights 10 and 12. Whole-image
integrity, processing robustness and statistical undetectability are not implied
by payload authentication.
