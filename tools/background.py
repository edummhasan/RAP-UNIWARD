#!/usr/bin/env python3
"""Linux background batch control, with a locked state file and PID reuse checks."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]  # field 22: process start time
    except (OSError, IndexError):
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'tail', 'stop'])
    args = parser.parse_args()
    os.chdir(ROOT)
    control = Path(os.environ.get('CONTROL_DIR', 'logs/background')).resolve()
    control.mkdir(parents=True, exist_ok=True)
    state_path = control / 'state.json'
    with (control / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        pid = state.get('pid', 0)
        alive = bool(pid and identity(pid) == state.get('starttime'))
        if args.action == 'start':
            if alive:
                raise SystemExit(f'Batch already running: pid={pid}')
            for name in ('RAP_ENC_KEY', 'RAP_XOR_KEY', 'RAP_STC_KEY'):
                if not os.environ.get(name):
                    raise SystemExit(f'Set {name} externally')
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            log = control / f'batch_{stamp}.log'
            with log.open('ab') as output:
                process = subprocess.Popen([str(ROOT / 'scripts/batch.sh')], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True)
            state = dict(pid=process.pid, starttime=identity(process.pid), log=str(log))
            temp = control / 'state.tmp'
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(state_path)
            print(f'Started pid={process.pid}; log={log}')
        elif args.action == 'status':
            print(f'RUNNING pid={pid}' if alive else 'STOPPED')
            if state.get('log'): print('log=' + state['log'])
        elif args.action == 'stop':
            if alive:
                if os.getpgid(pid) != pid:
                    raise SystemExit('Process is no longer the expected session leader; no signal sent')
                os.killpg(pid, signal.SIGTERM)
                print(f'Termination requested for batch pid={pid}')
            else: print('STOPPED')
        else:
            log = state.get('log')
            if not log or not Path(log).is_file(): raise SystemExit('No batch log exists')
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.execvp('tail', ['tail', '-n', os.environ.get('LINES', '100'), '-f', log])
    return 0

if __name__ == '__main__':
    sys.exit(main())
