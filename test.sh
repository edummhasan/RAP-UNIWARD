#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 tools/check_repository.py
python3 -m unittest discover -s src/test/python -v
./tools/run_native_tests.sh
python3 tools/run_cpu_tests.py "$@"
