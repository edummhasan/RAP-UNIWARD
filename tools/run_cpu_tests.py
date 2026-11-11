#!/usr/bin/env python3
"""CPU/JNI software checks. Fixtures are diagnostics, never thesis experiments.

Every Java test main is run with assertions enabled. GPU skips are recorded as
SKIP_OR_PARTIAL, never as accelerator passes. Relative test artifacts are kept
outside the source tree and removed after the corresponding test finishes.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-no-skips", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root / "build/test-results/java").resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(RAP_BACKEND="CPU", RAP_SI_BACKEND="CPU", RAP_BALANCE_PROFILE="B")
    java = env.get("JAVA_BIN", "java")
    classpath = os.pathsep.join(str(root / "build" / p)
                               for p in ("classes", "test-classes"))
    base = [java, "-ea", "-Xmx3g", "-Djava.awt.headless=true",
            "-Djava.library.path=" + str(root / "build/native"), "-cp", classpath]
    results = []
    for source in sorted((root / "src/test/java").rglob("*.java")):
        if not re.search(r"public\s+static\s+void\s+main\s*\(", source.read_text()):
            continue
        klass = ".".join(source.relative_to(root / "src/test/java").with_suffix("").parts)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rap-software-check-") as work:
            try:
                run = subprocess.run(base + [klass], cwd=work, env=env,
                                     capture_output=True, text=True, timeout=300)
                log = run.stdout + run.stderr
                exit_code = run.returncode
                status = ("BLOCKED_DEPENDENCY" if exit_code and not (root / "build/native/libjpegcore.so").is_file() and "no jpegcore in java.library.path" in log else
                          "FAIL" if exit_code else
                          "SKIP_OR_PARTIAL" if re.search(r"\bSKIP(?:PED)?\b", log, re.I)
                          else "PASS")
            except subprocess.TimeoutExpired as exc:
                text = exc.stdout or b""
                log = text.decode(errors="replace") if isinstance(text, bytes) else text
                log += "\nTIMEOUT after 300 seconds\n"
                exit_code, status = None, "TIMEOUT"
        (output / (source.stem + ".log")).write_text(log)
        results.append({"class": klass, "exit_code": exit_code, "status": status,
                        "seconds": round(time.monotonic() - start, 3),
                        "log": source.stem + ".log"})
        print(status + " " + klass, flush=True)
        if status in ("FAIL", "TIMEOUT", "BLOCKED_DEPENDENCY"):
            print(log[-2500:], flush=True)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    counts = dict(collections.Counter(row["status"] for row in results))
    summary = {"kind": "SOFTWARE_CHECKS_WITH_TEST_FIXTURES", "backend": "CPU",
               "assertions_enabled": True, "programs": len(results), "counts": counts,
               "gpu_qualification": "NOT_ESTABLISHED"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return int(not results or any(r["status"] in ("FAIL", "TIMEOUT", "BLOCKED_DEPENDENCY") for r in results)
               or (args.require_no_skips and counts.get("SKIP_OR_PARTIAL", 0) > 0))


if __name__ == "__main__":
    sys.exit(main())
