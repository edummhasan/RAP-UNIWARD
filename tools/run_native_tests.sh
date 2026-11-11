#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p build/test-results/native
for kind in self_test concurrency_test; do
  g++ -O3 -ffp-contract=off -std=c++17 -Wall -Wextra -Wpedantic -pthread \
    "src/test/native/native_stc_${kind}.cpp" \
    src/main/native/original_stc_adapter.cpp \
    src/main/native/original_stc/stc_embed_c.cpp \
    src/main/native/original_stc/stc_extract_c.cpp \
    src/main/native/original_stc/common.cpp \
    src/main/native/original_stc/coding_loss.cpp \
    -Isrc/main/native -Isrc/main/native/original_stc \
    -o "build/native_stc_${kind}"
  "build/native_stc_${kind}" | tee "build/test-results/native/native_stc_${kind}.log"
done
