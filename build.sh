#!/usr/bin/env bash
# Build RAP-UNIWARD Java classes, tests, executable JAR and native libraries.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
JAVA_ONLY=false
case "${1:-}" in
  '') ;;
  --java-only) JAVA_ONLY=true ;;
  -h|--help) echo 'Usage: ./build.sh [--java-only]'; echo 'RAP_BUILD_BACKEND=CPU|CUDA|HIP|ALL|AUTO; RAP_JAVA_RELEASE=22 (21 also supported).'; exit 0 ;;
  *) echo "Unknown build option: $1" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many build arguments' >&2; exit 2; }
MODE="${RAP_BUILD_BACKEND:-CPU}"; MODE="${MODE^^}"
[[ "$MODE" == ROCM ]] && MODE=HIP
case "$MODE" in CPU|CUDA|HIP|ALL|AUTO) ;; *) echo 'Invalid RAP_BUILD_BACKEND' >&2; exit 2;; esac
RELEASE="${RAP_JAVA_RELEASE:-22}"
[[ "$RELEASE" =~ ^[0-9]+$ && "$RELEASE" -ge 21 ]] || { echo 'RAP_JAVA_RELEASE must be at least 21' >&2; exit 2; }
JAVA_BIN="${JAVA_BIN:-java}"
command -v "$JAVA_BIN" >/dev/null || { echo 'Java is not installed' >&2; exit 2; }
# Some modular Java installations contain compiler tools without launcher scripts.
if command -v javac >/dev/null; then JAVAC=(javac); else JAVAC=("$JAVA_BIN" -m jdk.compiler/com.sun.tools.javac.Main); fi
if command -v jar >/dev/null; then JAR_TOOL=(jar); else JAR_TOOL=("$JAVA_BIN" -m jdk.jartool/sun.tools.jar.Main); fi
LANGUAGE_FLAGS=(--release "$RELEASE")
if [[ "$JAVA_ONLY" == true ]] && ! "${JAVAC[@]}" --help | grep -A1 'Supported releases:' | head -n 2 | grep -qw "$RELEASE"; then
  COMPILER_MAJOR="$("${JAVAC[@]}" -version 2>&1 | sed -E 's/^javac ([0-9]+).*/\1/')"
  [[ "$COMPILER_MAJOR" == "$RELEASE" ]] || { echo "Compiler $COMPILER_MAJOR cannot target Java $RELEASE" >&2; exit 2; }
  LANGUAGE_FLAGS=(-source "$RELEASE" -target "$RELEASE")
  echo 'Using matching compiler/runtime APIs for the Java-only check (ct.sym unavailable).'
fi
VERSION="$(<VERSION)"
[[ "$VERSION" == 1.0.0.1 ]] || { echo 'Unexpected VERSION file' >&2; exit 2; }
BUILD="$ROOT/build"; CLASSES="$BUILD/classes"; TEST_CLASSES="$BUILD/test-classes"; NATIVE="$BUILD/native"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"; NVCC="${NVCC:-$CUDA_ROOT/bin/nvcc}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"; HIPCC="${HIPCC:-$ROCM_ROOT/bin/hipcc}"
if [[ "$MODE" == AUTO ]]; then
  if [[ -x "$NVCC" ]]; then MODE=CUDA; elif [[ -x "$HIPCC" ]]; then MODE=HIP; else MODE=CPU; fi
fi
if [[ "$JAVA_ONLY" == false ]]; then
  command -v g++ >/dev/null || { echo 'Install a C++17 compiler' >&2; exit 2; }
  JAVA_PATH="$(readlink -f "$(command -v "$JAVA_BIN")")"
  JAVA_HOME="${JAVA_HOME:-$(cd "$(dirname "$JAVA_PATH")/.." && pwd)}"; export JAVA_HOME
  [[ -f "$JAVA_HOME/include/jni.h" ]] || { echo "JNI headers missing: install the JDK development package and set JAVA_HOME ($JAVA_HOME)" >&2; exit 2; }
  JPEG_INCLUDE_DIR="${JPEG_INCLUDE_DIR:-/usr/include}"
  [[ -f "$JPEG_INCLUDE_DIR/jpeglib.h" ]] || { echo 'Install libjpeg development headers, or set JPEG_INCLUDE_DIR' >&2; exit 2; }
  if [[ "$MODE" == CUDA || "$MODE" == ALL ]]; then
    [[ -x "$NVCC" ]] || { echo "nvcc is missing at $NVCC" >&2; exit 2; }
  fi
  if [[ "$MODE" == HIP || "$MODE" == ALL ]]; then
    [[ -x "$HIPCC" ]] || { echo "hipcc is missing at $HIPCC" >&2; exit 2; }
    [[ -n "${RAP_HIP_ARCH:-}" ]] || { echo 'Set RAP_HIP_ARCH to your actual GPU target (from rocminfo)' >&2; exit 2; }
  fi
fi
mkdir -p "$CLASSES" "$TEST_CLASSES" "$NATIVE" "$BUILD/jni-headers"
rm -rf "$CLASSES" "$TEST_CLASSES"
mkdir -p "$CLASSES" "$TEST_CLASSES"
echo "Building RAP-UNIWARD $VERSION: Java $RELEASE, backend $MODE, java_only=$JAVA_ONLY"
find src/main/java -name '*.java' -print | LC_ALL=C sort > "$BUILD/main-sources.txt"
"${JAVAC[@]}" "${LANGUAGE_FLAGS[@]}" -encoding UTF-8 -Xlint:all -d "$CLASSES" -h "$BUILD/jni-headers" @"$BUILD/main-sources.txt"
find src/test/java -name '*.java' -print | LC_ALL=C sort > "$BUILD/test-sources.txt"
"${JAVAC[@]}" "${LANGUAGE_FLAGS[@]}" -encoding UTF-8 -Xlint:all -cp "$CLASSES" -d "$TEST_CLASSES" @"$BUILD/test-sources.txt"
if [[ "$JAVA_ONLY" == false ]]; then
  rm -f "$NATIVE"/*.so
  CXX_FLAGS=(-O3 -ffp-contract=off -std=c++17 -Wall -Wextra -Wpedantic -fPIC)
  JNI_FLAGS=(-I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux")
  JPEG_LINK=(-ljpeg)
  [[ -z "${JPEG_LIB_DIR:-}" ]] || JPEG_LINK=(-L"$JPEG_LIB_DIR" -Wl,-rpath,"$JPEG_LIB_DIR" -ljpeg)
  g++ "${CXX_FLAGS[@]}" -shared src/main/c/jpegcore.cpp "${JNI_FLAGS[@]}" \
    -I"$JPEG_INCLUDE_DIR" "${JPEG_LINK[@]}" -o "$NATIVE/libjpegcore.so"
  g++ "${CXX_FLAGS[@]}" -shared src/main/native/stc_jni_bridge.cpp src/main/native/original_stc_adapter.cpp \
    src/main/native/original_stc/{stc_embed_c,stc_extract_c,common,coding_loss}.cpp \
    -Isrc/main/native -Isrc/main/native/original_stc "${JNI_FLAGS[@]}" -o "$NATIVE/liboriginalstcjni.so"
  if [[ "$MODE" == CUDA || "$MODE" == ALL ]]; then
    if [[ -n "${RAP_CUDA_ARCH:-}" ]]; then CUDA_FLAGS=(-arch="$RAP_CUDA_ARCH"); else
      CUDA_FLAGS=(-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
        -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90)
    fi
    "$NVCC" -O3 -std=c++17 "${CUDA_FLAGS[@]}" --fmad=false --prec-div=true --prec-sqrt=true \
      -Xcompiler=-ffp-contract=off -Xcompiler=-fPIC -shared src/main/cuda/{rap_si_cuda,rap_true_parallel_cuda,rap_stc_cuda}.cu \
      src/main/native/original_stc/common.cpp -Isrc/main/native -Isrc/main/native/original_stc "${JNI_FLAGS[@]}" \
      -Xlinker=-rpath -Xlinker="$CUDA_ROOT/lib64" -o "$NATIVE/librapcuda.so"
  fi
  if [[ "$MODE" == HIP || "$MODE" == ALL ]]; then
    "$HIPCC" --offload-arch="$RAP_HIP_ARCH" -O3 -ffp-contract=off -fno-fast-math -std=c++17 -fPIC -shared \
      src/main/hip/rap_si_hip.cpp "${JNI_FLAGS[@]}" -Wl,-rpath,"$ROCM_ROOT/lib" -o "$NATIVE/libraphip.so"
  fi
fi
"${JAR_TOOL[@]}" --create --file "$BUILD/rap-uniward-$VERSION.jar" --main-class thesis.rap.app.RapUniward -C "$CLASSES" .
"$JAVA_BIN" -jar "$BUILD/rap-uniward-$VERSION.jar" --version
if [[ "$JAVA_ONLY" == true ]]; then echo 'Java build passed; native libraries were not built or validated.'; else echo 'Build passed.'; fi
