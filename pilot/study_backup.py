"""Back up the live-study results: one compressed archive per finished batch, plus a SHA-256 manifest of every
file. Archives go to ~/Dropbox/P9_study_backup (meant to be set online-only); small state snapshots also to pilot/study_archive (inside
Dropbox). Big batch archives are not duplicated in Dropbox, whose raw jobs_study/ files are the second copy.
Manifests (small text) go to pilot/study_manifests and are committed to the local git repository.
Never overwrites an existing archive. Safe to rerun.

  python pilot/study_backup.py          # back up every finished batch not yet archived, plus ledger/logs/secrets
  python pilot/study_backup.py --watch  # keep doing that every 2 minutes until the study ends
"""
import hashlib
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs_study"
DESTS = [Path.home() / "Dropbox" / "P9_study_backup", ROOT / "study_archive"]
MANI = ROOT / "study_manifests"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive(name, paths):
    """Write name.tar.gz to the first destination, verify it, copy to the others; skip if it already exists."""
    files = sorted(f for p in paths if p.exists() for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())
    if not files:
        return False
    MANI.mkdir(exist_ok=True)
    mani = MANI / f"{name}.sha256"
    for d in DESTS:
        d.mkdir(parents=True, exist_ok=True)
    first = DESTS[0] / f"{name}.tar.gz"
    if not first.exists():
        tmp = first.with_suffix(".part")
        with tarfile.open(tmp, "w:gz") as t:
            for f in files:
                t.add(f, arcname=str(f.relative_to(ROOT)))
        with tarfile.open(tmp, "r:gz") as t:          # verify the archive reads back with every file
            assert len([m for m in t.getmembers() if m.isfile()]) == len(files)
        tmp.rename(first)
        mani.write_text("".join(f"{sha(f)}  {f.relative_to(ROOT)}\n" for f in files))
        print(time.strftime("%H:%M:%S"), f"archived {name}: {len(files)} files, {first.stat().st_size / 1e6:.0f} MB", flush=True)
    for d in DESTS[1:]:
        if name.startswith("batch") and not name.endswith("aborted_attempt"):
            continue   # big batch archives only outside Dropbox: the raw files in Dropbox are the second copy (disk)
        if not (d / first.name).exists():
            shutil.copy2(first, d / first.name)
    if name.startswith("secret") or name.startswith("state"):
        for d in DESTS:
            (d / first.name).chmod(0o600)
    return True


def once():
    for marker in sorted(JOBS.glob(".batch??_done")):
        b = marker.name[6:8]
        archive(f"batch{b}", [marker] + sorted(JOBS.glob(f"openhands/trial-*/b{b}_*"))
                + sorted(ROOT.glob(f"logs_study_b{b}_*.txt")))
    for marker in sorted(JOBS.glob(".batch??_rerun_done")):     # batches 4 and 6 were refused whole and rerun
        b = marker.name[6:8]
        archive(f"batch{b}_rerun", [marker] + sorted(JOBS.glob(f"openhands/trial-*/b{b}_*"))
                + sorted(ROOT.glob(f"logs_study_b{b}_*.txt")))
    for marker in sorted(JOBS.glob(".sec??_done")):            # secondary run (usage-limit blocked tasks)
        b = marker.name[4:6]
        archive(f"secondary{b}", [marker] + sorted(JOBS.glob(f"openhands/trial-*/sec{b}_*"))
                + sorted(ROOT.glob(f"logs_study_sec{b}_*.txt")))
    aborted = ROOT / "jobs_study_aborted"
    archive("batch02_aborted_attempt", [aborted])
    # ledger, driver log, order and study secrets change over time: snapshot with a timestamp when they change
    state = [ROOT / "ledger.jsonl", ROOT / "logs_study_driver.txt", ROOT / "secret"]
    digest = hashlib.sha256("".join(sha(f) for p in state if p.exists()
                                    for f in ([p] if p.is_file() else sorted(p.rglob("*"))) if f.is_file()).encode()).hexdigest()[:12]
    if not any(DESTS[0].glob(f"state_*_{digest}.tar.gz")):
        archive(f"state_{time.strftime('%Y%m%dT%H%M%S')}_{digest}", state)


if __name__ == "__main__":
    once()
    if "--watch" in sys.argv:
        while True:
            log = (ROOT / "logs_study_driver.txt").read_text()
            if "SECONDARY DONE" in log or "SECONDARY STOP" in log:
                time.sleep(60)
                once()
                print("backup watcher: study ended, final backup done", flush=True)
                break
            time.sleep(120)
            once()
