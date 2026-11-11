"""Repository workflow tests; these fixtures are not experimental observations."""
import csv
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[3]

class ManifestMergeTest(unittest.TestCase):
    def write(self, path, indices):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['pair_index','total_pairs','status','message_name'])
            writer.writeheader()
            for index in indices:
                writer.writerow(dict(pair_index=index,total_pairs=2,status='COMPLETED',message_name='0001.msg'))

    def run_merge(self, output, *inputs):
        env=os.environ.copy(); env.pop('RAP_ALLOW_INCOMPLETE_MERGE',None)
        return subprocess.run(['python3',str(ROOT/'tools/merge_manifests.py'),str(output),*map(str,inputs)],env=env,capture_output=True,text=True)

    def test_complete_merge_preserves_text(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d); a=directory/'a.csv'; b=directory/'b.csv'; out=directory/'out.csv'
            self.write(a,[2]); self.write(b,[1])
            self.assertEqual(self.run_merge(out,a,b).returncode,0)
            with out.open(newline='') as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['pair_index'] for r in rows],['1','2'])
            self.assertEqual(rows[0]['message_name'],'0001.msg')

    def test_missing_row_never_publishes(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d); a=directory/'a.csv'; out=directory/'out.csv'
            self.write(a,[1]); out.write_text('existing result\n')
            self.assertNotEqual(self.run_merge(out,a).returncode,0)
            self.assertEqual(out.read_text(),'existing result\n')

    def test_duplicate_never_publishes(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d); a=directory/'a.csv'; b=directory/'b.csv'; out=directory/'out.csv'
            self.write(a,[1]); self.write(b,[1])
            self.assertNotEqual(self.run_merge(out,a,b).returncode,0)
            self.assertFalse(out.exists())

class BatchControlTest(unittest.TestCase):
    def test_invalid_gpu_lists_rejected_before_launch(self):
        for value in ('', '0,', ',0', '0,,1', '-1', '0,0'):
            with self.subTest(value=value):
                env=os.environ.copy() | {'RAP_BACKEND':'CUDA', 'GPU_IDS':value}
                result=subprocess.run(['bash',str(ROOT/'scripts/multi-gpu.sh')],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,2)
                self.assertIn('GPU_IDS',result.stderr)

    def test_termination_stops_running_shards(self):
        # Real coordinator, diagnostic child processes; no GPU algorithm is run.
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            scripts=directory/'scripts'; scripts.mkdir()
            (scripts/'multi-gpu.sh').write_bytes((ROOT/'scripts/multi-gpu.sh').read_bytes())
            child=scripts/'batch.sh'
            child.write_text('''#!/usr/bin/env python3
import os, signal
from pathlib import Path
root=Path(os.environ['RUN_ROOT'])
pid=os.getpid()
def stop(*args):
    (root / f'stopped-{pid}').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
(root / f'ready-{pid}').touch()
while True: signal.pause()
''')
            child.chmod(0o755)
            run=directory/'run'
            env=os.environ.copy() | {'RAP_BACKEND':'CUDA','GPU_IDS':'0,1','RUN_ROOT':str(run),
                'RAP_ENC_KEY':'fixture-encryption','RAP_XOR_KEY':'fixture-xor','RAP_STC_KEY':'fixture-stc',
                'DELETE_JPEGS_AFTER_EVALUATION':'false','START_INDEX':'1'}
            process=subprocess.Popen(['bash',str(scripts/'multi-gpu.sh')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                deadline=time.monotonic()+10
                while len(list(run.glob('ready-*')))<2 and process.poll() is None and time.monotonic()<deadline:
                    time.sleep(0.02)
                self.assertEqual(len(list(run.glob('ready-*'))),2,'Coordinator did not launch both test children')
                process.send_signal(signal.SIGTERM)
                out,err=process.communicate(timeout=5)
                self.assertEqual(process.returncode,143,(out,err))
                deadline=time.monotonic()+5
                while len(list(run.glob('stopped-*')))<2 and time.monotonic()<deadline:
                    time.sleep(0.02)
                self.assertEqual(len(list(run.glob('stopped-*'))),2,'Running child was left behind')
            finally:
                if process.poll() is None:
                    process.kill(); process.communicate(timeout=5)
                for ready in run.glob('ready-*'):
                    if not (run/ready.name.replace('ready-','stopped-')).exists():
                        try: os.kill(int(ready.name.split('-')[1]),signal.SIGTERM)
                        except ProcessLookupError: pass

if __name__=='__main__': unittest.main()
