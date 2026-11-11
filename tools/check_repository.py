#!/usr/bin/env python3
"""Check release naming, script syntax, package paths and generated JNI exports."""
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
errors=[]
def require(condition, message):
    if not condition: errors.append(message)
require((ROOT/'VERSION').read_text().strip()=='1.0.0.1','VERSION must be 1.0.0.1')
allowed_domains=['AHP-PAYLOAD-AAD-V14.2.0-GEOM-V1','AHP-PAYLOAD-AAD-V14.2.0','ahp-order-v1','ahp-direction-v1']
for p in (ROOT/'src').rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc', '.pyo'): continue
    try:
        content=p.read_text(encoding='utf-8')
    except UnicodeError:
        require(False, f'Unexpected binary source file: {p.relative_to(ROOT)}')
        continue
    for domain in allowed_domains: content=content.replace(domain,'')
    require(not re.search(r'\bAHP\b|thesis[._/]ahp\b|\bV1[0-9](?:[._]|\b)|_v1[0-9]_',content),f'Stale identity in {p.relative_to(ROOT)}')
    if p.suffix=='.java':
        match=re.search(r'^package\s+([\w.]+);',content,re.M)
        require(bool(match),f'Package missing: {p.name}')
        if match: require('/'.join(match[1].split('.'))==str(p.parent.relative_to(ROOT/'src'/('test' if '/test/' in str(p) else 'main')/'java')),f'Package path mismatch: {p.name}')
for p in ROOT.rglob('*.sh'):
    result=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
    require(result.returncode==0,f'Shell syntax: {p.relative_to(ROOT)}: {result.stderr}')
    require(bool(p.stat().st_mode & 0o111),f'Executable bit missing: {p.relative_to(ROOT)}')
headers=list((ROOT/'build/jni-headers').glob('*.h'))
if headers:
    declarations=set()
    for p in headers: declarations.update(re.findall(r'JNICALL\s+(Java_\w+)',p.read_text()))
    exports=set()
    for p in (ROOT/'src/main').rglob('*'):
        if p.suffix in ('.cpp','.cu'): exports.update(re.findall(r'JNICALL\s+(Java_\w+)',p.read_text()))
    require(declarations<=exports,'Missing JNI definitions: '+', '.join(sorted(declarations-exports)))
    print(f'JNI declarations matched: {len(declarations)}')
else:
    print('JNI export comparison not run: build Java sources first')
for rel in ('README.md','docs/RUNTIME.md','docs/THESIS_ALIGNMENT.md','build.sh','run.sh','test.sh','scripts/batch.sh','.gitignore'):
    require((ROOT/rel).is_file(),f'Required file missing: {rel}')
for error in errors: print('FAIL '+error,file=sys.stderr)
print(f'Repository checks: {"PASS" if not errors else "FAIL"}')
sys.exit(bool(errors))
