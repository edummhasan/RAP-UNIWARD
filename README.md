# RAP-UNIWARD

**Version 1.0.0.1**

Reference-Assisted Probability-Aware UNIWARD: adaptive JPEG steganography for
normalized 24-bit RGB images, with quality-factor selection and pre-embedding
statistical risk modeling. Sender and receiver share the same normalized PNG
source and three independent secret inputs.

The implementation modifies Cb/Cr coefficients in zero-based zig-zag ranks
16–47, including zero-valued reference coefficients. It independently scans
QF 75–100 for each requested sampling geometry and accepts an output only after
coefficient write-back checks, authenticated extraction and exact message equality.
The seven supported geometries are 4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, 4:4:1 and
4:1:0. STC height 10 is the default; 12 is a separate supported profile.

## Requirements

- Linux, Bash, Python 3, a JDK with compiler/JNI development files, and a C++17 compiler.
- Java 22 is the default build target. Java 21 is supported by setting `RAP_JAVA_RELEASE=21`.
- libjpeg development headers and library: `libjpeg-dev` on Debian/Ubuntu, or
  `libjpeg-turbo-devel` on Rocky/RHEL. The runtime library alone is insufficient.
- Optional CUDA or ROCm/HIP development toolkit for GPU builds.

On an Ubuntu host, the CPU prerequisites can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk-headless g++ libjpeg-dev python3
RAP_JAVA_RELEASE=21 RAP_BUILD_BACKEND=CPU ./build.sh
./test.sh
```

With an installed JDK 22 and `JAVA_HOME`/`PATH` pointing to it:

```bash
RAP_BUILD_BACKEND=CPU ./build.sh
./test.sh
./run.sh --version
./run.sh --help
```

The JAR is `build/rap-uniward-1.0.0.1.jar`. `build/native/` contains the host-built
JNI libraries. This is a self-contained source repository; it still needs the
listed compiler, runtime and JPEG dependencies.

## Inputs and keys

Put your PNG files in `input/` and arbitrary binary message files in `msg/`.
The default batch extension is `.msg`. Keep image and message identifiers as text;
`000001` and `0001` are distinct. Dataset files and outputs are excluded from Git.

Generate independent random secrets, store them securely for later extraction,
and export them before running. For example:

```bash
export RAP_ENC_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export RAP_XOR_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export RAP_STC_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

Do not regenerate the keys between embedding and extraction. Nonblank/distinct
validation does not establish key entropy. Do not put real secrets in tracked files.

## Single image and message

```bash
RAP_BACKEND=CPU ./run.sh embed \
  --png input/image.png --msgFile msg/001.msg --out output/stego.jpg \
  --geometryMode 420 --stcHeight 10 \
  --encKeyEnv RAP_ENC_KEY --xorKeyEnv RAP_XOR_KEY --stcKeyEnv RAP_STC_KEY

RAP_BACKEND=CPU ./run.sh extract \
  --png input/image.png --stego output/stego.jpg --outFile output/recovered.msg \
  --geometryMode 420 --stcHeight auto \
  --encKeyEnv RAP_ENC_KEY --xorKeyEnv RAP_XOR_KEY --stcKeyEnv RAP_STC_KEY

cmp msg/001.msg output/recovered.msg
```

Paths passed through the supplied scripts are relative to the repository root.
For a subset, use `--geometryMode 444,420,411`; for all seven, use
`--geometryMode ALL_GEOMETRY`. A multiple-geometry single-pair request adds
`_g444`, `_g420`, etc. to the requested output stem.

## Batch execution

```bash
RAP_BACKEND=CPU GEOMETRIES=ALL_GEOMETRY ./scripts/batch.sh
```

Each discovered image is paired with every discovered message and each selected
geometry. The default is virtual-thread execution, `WORKERS=auto`, and CPU slots
based on `nproc`. Workload and memory limits must suit the host. A conservative
configuration is in `config/cpu.env.example`:

```bash
cp config/cpu.env.example config/local.env
# Edit config/local.env, then load it:
source config/local.env
./scripts/batch.sh
```

For a 28-core CPU allocation, after selecting suitable memory limits:

```bash
export RAP_BACKEND=CPU WORKERS=28 CPU_SLOTS=28 CARRIER_PARALLELISM=28
export RAP_PREPARED_CACHE_STATES=8 RAP_SOURCE_CACHE_IMAGES=1
export RAP_QF_PREFETCH_DEPTH=0 RAP_QF_PREFETCH_WORKERS=1
./scripts/background.sh start
./scripts/background.sh status
./scripts/background.sh tail
./scripts/background.sh stop
```

Set `CPU_AFFINITY` only to CPU IDs available to the process; CPU count alone does
not establish which IDs are available. The wrapper applies affinity with `taskset`.
A stopped run can be resumed with `RESUME=true ./scripts/batch.sh`. Resume verifies
the JPEG and its sidecar, then authenticates the recovered message. An unverified
existing output is quarantined before a replacement attempt.

## Results

The default `RESULT_ONLY` mode writes `results/audit_final_qf.csv`: a terminal
record for each executed image–message–geometry condition, including failure
reason and final-QF state. Successful rows include aggregate quality, STC,
coefficient and Cb/Cr statistics. Failed rows leave unavailable measurements blank.
Rerunning a condition appends another record; reconciliation is a separate analysis.

`output/` holds accepted stego JPEGs and `.verified` integrity sidecars.
`results/batch_manifests/` holds operational batch-status CSVs. Set `RAP_REPORT_DIR`
to redirect ledgers and audit artifacts. `QF_FORENSIC` additionally records QF
trials and audit bundles. See [runtime reference](docs/RUNTIME.md) for exact paths,
mode-dependent outputs and all configuration controls.

The `summarize` command consumes the `evaluation_*.csv` tables produced in
`QF_FORENSIC` mode. It does **not** consume the default terminal ledger or reproduce
the thesis's archive reconciliation and clustered bootstrap analysis.

## GPU execution

```bash
# NVIDIA: choose the architecture supported by your actual GPU and toolkit.
RAP_BUILD_BACKEND=CUDA RAP_CUDA_ARCH=sm_90 ./build.sh
RAP_BACKEND=CUDA GEOMETRIES=ALL_GEOMETRY ./scripts/batch.sh

# AMD: replace the target with the value reported by rocminfo on your host.
RAP_BUILD_BACKEND=HIP RAP_HIP_ARCH=gfx1103 ./build.sh
RAP_BACKEND=HIP ./scripts/batch.sh

# Multiple NVIDIA devices; one process per GPU:
RAP_BACKEND=CUDA GPU_IDS=0,1 ./scripts/multi-gpu.sh
# Multiple AMD devices:
RAP_BACKEND=HIP GPU_IDS=0,1 ./scripts/multi-gpu.sh
```

CUDA supports multiple pipeline stages. HIP accelerates SI cost evaluation; the
other stages use CPU implementations. Explicit GPU modes fail closed unless
`RAP_ACCELERATOR_FALLBACK=true` is selected. `AUTO` permits CPU fallback. Separate
GPU runs need real device qualification and CPU/GPU extraction checks; CPU test
results do not establish GPU correctness. Multi-GPU reports are isolated per shard.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/main/java/thesis/rap/` | Embedding, extraction, image preparation, metrics and runtime |
| `src/main/c/` | JPEG coefficient JNI layer |
| `src/main/native/` | Binary64 STC and JNI adapter |
| `src/main/cuda/`, `src/main/hip/` | Optional accelerator implementations |
| `src/test/` | Java, native and Python software checks |
| `scripts/` | Batch, background and multi-GPU entry points |
| `tools/` | Test runner, manifest merger and thesis/evidence checks |
| `config/` | Editable configuration example |
| `input/`, `msg/` | User-supplied data; ignored except directory placeholders |
| `output/`, `results/`, `logs/` | Runtime artifacts; ignored except placeholders |
| `build/` | Generated classes, JAR, native libraries, cache and test logs |
| `docs/` | Runtime contract and current thesis-alignment review |

## Validation and research scope

The packaging-host check compiled all production and test Java sources using
Java 21. Thirty Java test programs passed; two recorded accelerator skips/partial
coverage, and three JPEG integration programs were blocked by missing JNI/libjpeg
development files. Native STC self-tests and 64 concurrent round trips passed.
Five Python checks passed for manifest merging, device-list validation and stopping
active shard processes. The process-control checks use diagnostic children, not GPUs.
All 30 generated JNI declarations match their native source export names.
Java 22, full JPEG integration, CUDA and HIP execution remain target-host checks.
The included CI workflow builds and tests CPU configurations with Java 21 and 22;
it is a workflow definition, not evidence of a completed CI run.

[Thesis alignment and evidence review](docs/THESIS_ALIGNMENT.md) maps all six
algorithms to implementation sites, reconciles the supplied summaries and records
unresolved evidence issues. No dataset observations are created by the test fixtures.
PSNR/SSIM and successful extraction do not establish steganographic security.

The thesis specifies fixed authentication and ordering domain bytes. They are
preserved as protocol constants, independent of the application release number.
There are no prior release packages, changelogs, production logs or stored secrets
in this repository.

## Development and Git publication

Run `./build.sh` and `./test.sh` before accepting a change. Keep the codec,
framing, candidate order and numerical conventions synchronized at both endpoints.
Never weaken authentication or replace a failed check with a fabricated metric.
Tests use diagnostic fixtures; experiments require the actual corpus.

```bash
git init -b main
git add .
git status --short
git commit -m "Initial RAP-UNIWARD 1.0.0.1 source"
# Add your own remote and push when ready.
```

Inspect staged paths before committing. A distribution license was not supplied
with the source; `NOTICE` records this and the bundled STC attribution boundary.
No license or repository owner is invented here.
