#!/usr/bin/env python3
"""Check thesis references and reconcile supplied aggregate CSV evidence.

Identifiers remain strings. This does not infer condition-level validity from
summaries, rerun an experiment, or reconstruct unavailable raw observations.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


def read_rows(folder, filename):
    with (folder / filename).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--thesis-dir', type=Path, required=True)
    ap.add_argument('--evidence-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('results/thesis_check.json'))
    args = ap.parse_args()
    labels, refs = defaultdict(list), defaultdict(list)
    citations, bibkeys = set(), []
    inventory = []
    for path in sorted(args.thesis_dir.rglob('*')):
        if path.suffix not in ('.tex', '.bib'): continue
        raw = path.read_bytes()
        inventory.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest()))
        content = re.sub(r'(?<!\\)%[^\n]*', '', raw.decode('utf-8'))
        if path.suffix == '.bib':
            bibkeys.extend(re.findall(r'@\w+\s*\{\s*([^,\s]+)', content))
            continue
        for key in re.findall(r'\\label\{([^}]+)\}', content): labels[key].append(path.name)
        for key in re.findall(r'\\(?:eqref|ref|autoref|pageref|cref|Cref)\{([^}]+)\}', content): refs[key].append(path.name)
        for group in re.findall(r'\\(?:paren|text|auto|foot)?cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', content):
            citations.update(map(str.strip, group.split(',')))
    latex = dict(tex_files=sum(p.suffix == '.tex' for p in args.thesis_dir.rglob('*')),
                 labels=len(labels), reference_keys=len(refs), citation_keys=len(citations),
                 bibliography_entries=len(bibkeys), undefined_references={k:v for k,v in refs.items() if k not in labels},
                 duplicate_labels={k:v for k,v in labels.items() if len(v)>1},
                 missing_citations=sorted(citations-set(bibkeys)),
                 duplicate_bibkeys=[k for k,n in Counter(bibkeys).items() if n>1])
    directory=args.evidence_dir
    tables={p.name:read_rows(directory,p.name) for p in sorted(directory.glob('*.csv'))}
    evidence_inventory=[dict(file=p.name, rows=len(tables[p.name]), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in sorted(directory.glob('*.csv'))]
    checks=[]
    def check(name, condition, **detail):
        checks.append(dict(check=name, status='PASS' if condition else 'FAIL', **detail))
    raw={r['source_archive']:int(r['raw_rows']) for r in tables['raw_archive_counts.csv']}
    retained={r['source_archive']:int(r['selected_rows']) for r in tables['selected_archive_counts.csv']}
    superseded=sum(int(r['superseded_rows']) for r in tables['superseded_outcomes.csv'])
    outcomes=tables['outcome_reasons.csv']; total=sum(int(r['count']) for r in outcomes)
    successes=sum(int(r['count']) for r in outcomes if r['status']=='VERIFIED_SUCCESS')
    failures=total-successes
    check('archive_reconciliation',sum(raw.values())-sum(retained.values())==superseded and sum(retained.values())==total,
          raw=sum(raw.values()), retained=sum(retained.values()), superseded=superseded)
    check('condition_grid',total==3550*15*7,total=total,verified=successes,no_output=failures,
          success_percent=100*successes/total)
    for name,expected_rows,expected_trials in [('sampling_geometry_summary.csv',7,53250),('message_id_text_summary.csv',15,24850),('sampling_geometry_message_id_text_summary.csv',105,3550)]:
        rows=tables[name]
        check(name,len(rows)==expected_rows and sum(int(r['verified']) for r in rows)==successes and
              sum(int(r['no_output']) for r in rows)==failures and all(int(r['trials'])==expected_trials and
              int(r['verified'])+int(r['no_output'])==int(r['trials']) for r in rows))
    images=tables['image_success_counts.csv']
    check('per_image_grid',len(images)==3550 and len({r['image_id_text'] for r in images})==3550 and
          all(int(r['count'])==105 for r in images) and sum(int(r['sum']) for r in images)==successes)
    messages=tables['message_identity.csv']
    check('message_identities_and_framing',len(messages)==15 and len({r['message_sha256'] for r in messages})==15 and
          all(int(r['framed_bits'])==8*(35+int(r['raw_bytes'])) for r in messages),
          frame_overhead_bytes=35, wire_header_bytes=26, plaintext_metadata_bytes=9)
    qf_name=next(n for n in tables if re.fullmatch(r'qf_distribution(?:\(\d+\))?\.csv',n))
    qfs=tables[qf_name]
    good=[r for r in qfs if r['status']=='VERIFIED_SUCCESS']
    check('qf_population',sum(int(r['count']) for r in qfs)==total and sum(int(r['count']) for r in good)==successes and
          all(75<=int(r['terminal_qf'])<=100 for r in qfs),minimum=min(int(r['terminal_qf']) for r in good),maximum=max(int(r['terminal_qf']) for r in good))
    inv=tables['invariant_checks.csv']
    check('reported_invariants',len(inv)==37 and all(int(r['failed_or_missing'])==0 and int(r['applicable'])==int(r['passed']) for r in inv),
          categories=len(inv),qualification='Aggregate declarations, not independent replay')
    boot=tables['cluster_bootstrap_intervals.csv']
    distributions={r['']:r for r in tables['successful_numeric_distribution.csv']}
    def close(a,b): return abs(float(a)-float(b))<=1e-9*max(1,abs(float(a)),abs(float(b)))
    mean_checks=[]
    for row in boot:
        if row['metric']=='success_rate':
            mean_checks.append(close(row['mean'],successes/total))
        elif row['metric'] in distributions:
            mean_checks.append(close(row['mean'],distributions[row['metric']]['mean']))
    check('bootstrap_point_estimates_match',all(mean_checks),compared=len(mean_checks))
    check('bootstrap_intervals_contain_estimate',all(float(r['ci_low'])<=float(r['mean'])<=float(r['ci_high']) for r in boot))
    conflicts=[]; seen=set()
    for name,rows in tables.items():
        if not name.startswith('extremes_'): continue
        for r in rows:
            pointer=(r['source_archive'],r['source_row'],r['run_id'])
            if r['source_archive'] in raw and int(r['source_row'])>raw[r['source_archive']] and pointer not in seen:
                seen.add(pointer)
                conflicts.append(dict(file=name,source_archive=r['source_archive'],source_row=r['source_row'],
                    raw_rows=raw[r['source_archive']],image_id_text=r['image_id_text'],message_id_text=r['message_id_text'],run_id=r['run_id']))
    findings=dict(source_row_pointer_conflicts=conflicts,
        bootstrap_resamples_and_seed_present=any(re.search(r'resample|seed|random_state',key,re.I) for r in boot for key in r),
        exact_executed_source_identity_present=any(re.search(r'application_version|commit|source_sha',key,re.I) for rows in tables.values() for r in rows[:1] for key in r),
        evidence_scope='Summary tables only; no complete terminal ledger, original PNG/message corpus or serialized reference/stego population supplied')
    result=dict(latex=latex,evidence_checks=checks,evidence_findings=findings,thesis_inventory=inventory,evidence_inventory=evidence_inventory)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(latex=latex,checks=[{k:v for k,v in c.items() if k in ('check','status','raw','retained','superseded','total','verified','no_output','success_percent')} for c in checks],findings=findings),indent=2))
    return int(any(c['status']=='FAIL' for c in checks) or any(latex[k] for k in ('undefined_references','duplicate_labels','missing_citations','duplicate_bibkeys')))

if __name__=='__main__': raise SystemExit(main())
