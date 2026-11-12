#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VERSION="$(<VERSION)"
JAR="${RAP_JAR:-$ROOT/build/rap-uniward-$VERSION.jar}"
[[ -f "$JAR" ]] || { echo "Build first: ./build.sh (missing $JAR)" >&2; exit 2; }
# Split JVM options without evaluating shell syntax or expanding glob patterns.
read -r -a JVM_ARGS <<< "${JAVA_OPTS:--Djava.awt.headless=true}"
exec "${JAVA_BIN:-java}" "${JVM_ARGS[@]}" -Djava.library.path="$ROOT/build/native" -jar "$JAR" "$@"
