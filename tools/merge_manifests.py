#!/usr/bin/env python3
"""Validate full-grid shard manifests and atomically publish a merged CSV."""
import csv
import os
from pathlib import Path
import sys
import tempfile

if len(sys.argv)<3: raise SystemExit('usage: merge_manifests.py OUTPUT.csv SHARD1.csv [SHARD2.csv ...]')
out=Path(sys.argv[1]); rows={}; header=None; totals=set()
for item in sys.argv[2:]:
    path=Path(item)
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if header is None: header=reader.fieldnames
        elif reader.fieldnames!=header: raise SystemExit(f'Header mismatch: {path}')
        for row in reader:
            idx=int(row['pair_index']); total=int(row['total_pairs'])
            if idx<1 or idx>total: raise SystemExit(f'Invalid pair index {idx} of {total}')
            if idx in rows: raise SystemExit(f'Duplicate terminal pair index {idx}')
            if row['status']=='FAILED_EXCEPTION': raise SystemExit(f'Unexpected failure at pair {idx}')
            rows[idx]=row; totals.add(total)
if not rows or len(totals)!=1: raise SystemExit('Manifests must contain records with one consistent total_pairs value')
total=totals.pop()
missing_count=total-len(rows)
if missing_count and os.environ.get('RAP_ALLOW_INCOMPLETE_MERGE','false').lower()!='true':
    raise SystemExit(f'Incomplete merge: missing {missing_count} of {total} pairs')
out.parent.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile('w',newline='',encoding='utf-8',dir=out.parent,delete=False) as f:
    tmp=Path(f.name)
    writer=csv.DictWriter(f,fieldnames=header); writer.writeheader()
    for idx in sorted(rows): writer.writerow(rows[idx])
try: tmp.replace(out)
finally: tmp.unlink(missing_ok=True)
print(f'merged_rows={len(rows)} expected_pairs={total} missing_pairs={missing_count} output={out}')
