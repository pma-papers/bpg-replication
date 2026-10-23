"""Run a stage of the wear-allowance study with a process pool; completed runs are skipped."""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="campaign"); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--arms", nargs="*"); args = ap.parse_args()
protocol = json.loads((ROOT / "protocol.json").read_text())
arms = args.arms or list(protocol["arms"])
jobs = [(a, s) for a in arms for s in protocol["stages"][args.stage]["seeds"]]
def work(job):
    a, s = job
    p = subprocess.run([sys.executable, str(ROOT / "train.py"), "--arm", a, "--seed", str(s), "--stage", args.stage], capture_output=True, text=True)
    return (p.stdout.strip().splitlines() or [""])[-1] if p.returncode == 0 else f"FAILED {a} {s}: {p.stderr[-400:]}"
with ThreadPoolExecutor(args.workers) as ex:
    for f in as_completed([ex.submit(work, j) for j in jobs]):
        print(f.result(), flush=True)
