"""
Resumable training matrix (Colab: notebooks/colab_train.ipynb). Run from ml/:

    python train/matrix.py --configs configs/matrix/yolo26n_def3_fold0.yaml ... \
        --drive /content/drive/MyDrive/ZanderML --device 0 \
        [--set KEY=VALUE ...] [--profile auto|L4|A100|T4|none] [--max-attempts 2] [--dry-run]

Drive layout (docs/CONTRACTS.md §8.5), under --drive:
    bundles/<run>_bundle.zip + .sha256   finished marker (the .sha256 is written last)
    mirror/<run>/                        resume state, deleted once the bundle is safe
    logs/<run>.log                       full training output (appended per attempt)
    matrix_log.jsonl                     one line per started/finished/skipped attempt

Per config, in the given (priority) order:
  1. skip when bundles/<run>_bundle.zip has a matching .sha256 — even if its
     config_sha256 differs from today's effective config (warned): results are
     never overwritten. A zip without a valid marker is moved aside and redone. If Drive cannot be READ (an I/O error, not a bad
     bundle) the run is reported as failed and left alone: never moved aside,
     never retrained.
  2. run train/dispatch.py as a SUBPROCESS with --mirror-dir <drive>/mirror
     (the runner restores mirror/<run>/ into its run_dir and resumes), output
     teed to logs/<run>.log; a failure is retried up to --max-attempts times
     (not for exit code 2 = configuration/data error) and never stops the loop.
  3. on success copy the bundle, verify its sha, write the .sha256, then
     delete mirror/<run>/.
Drive trouble never stops a healthy run: a failing logs/<run>.log or
matrix_log.jsonl write is warned about once and the training goes on; an
error in the driver around one run fails that run, not the loop.
After every run: an ETA from the measured minutes/epoch (per family, else per
architecture; seeded from earlier sessions' matrix_log.jsonl). At the end: a
summary table; exit 1 if any run failed, 2 on config errors, 130 on Ctrl-C.

GPU profiles change only non-semantic runtime knobs — dataloader workers, and
cache=disk on hosts with <= 4 vCPUs — never batch/imgsz/epochs (those are the
config, and they are what the bundle's config_sha256 covers).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ML_DIR = Path(__file__).resolve().parents[1]
_TRAIN_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _TRAIN_DIR]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from train import bundle as B  # noqa: E402
from train import dispatch  # noqa: E402

PROFILES = {  # max dataloader workers per GPU class (bounded by the host's vCPUs)
    "A100": 12,
    "L4": 8,
    "T4": 4,
    "generic": 8,
}
PROFILE_CHOICES = ("auto", "L4", "A100", "T4", "none")
DISK_CACHE_MAX_CPUS = 4
TAIL_BYTES = 64 * 1024
TAIL_LINES = 40
ERROR_TAIL_CHARS = 4000


def utc_now() -> str:
    return B.utc_now()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------- profiles


def detect_gpu() -> Optional[str]:
    """Name of GPU 0 via nvidia-smi (no torch import in the driver), or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return names[0] if names else None


def profile_for(requested: str, gpu_name: Optional[str]) -> Optional[str]:
    if requested == "none":
        return None
    if requested != "auto":
        return requested
    if not gpu_name:
        return None
    upper = gpu_name.upper()
    for key in ("A100", "L4", "T4"):
        if key in upper:
            return key
    return "generic"


def runtime_knobs(profile: Optional[str], cpu_count: Optional[int]) -> dict:
    """{"workers": n, "cache": "disk"} — never anything that changes the model."""
    if profile is None:
        return {}
    cpus = max(1, int(cpu_count or 1))
    knobs = {"workers": max(1, min(PROFILES[profile], cpus))}
    if cpus <= DISK_CACHE_MAX_CPUS:
        knobs["cache"] = "disk"  # JPEG decode is the bottleneck on few vCPUs; disk cache stays deterministic
    return knobs


# ------------------------------------------------------------------- items


@dataclass
class Item:
    config: Path
    name: str
    family: str
    arch: str
    epochs: int
    config_sha: Optional[str]


@dataclass
class RunRequest:
    item: Item
    attempt: int
    cmd: list
    env: dict
    cwd: Path
    log_path: Path
    mirror_dir: Path
    run_root: Path


@dataclass
class LaunchResult:
    returncode: int
    tail: str


@dataclass
class Outcome:
    name: str
    status: str                      # ok | failed | skipped
    tries: int = 0
    resumed: Optional[bool] = None
    minutes: float = 0.0
    epochs_completed: Optional[int] = None
    note: str = ""
    records: list = field(default_factory=list)


Launcher = Callable[[RunRequest], LaunchResult]


def resolve_config_path(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    if (ML_DIR / p).is_file():
        return (ML_DIR / p).resolve()
    raise B.ConfigError(f"config not found: {path}")


def load_items(configs: list, overrides: dict) -> tuple[list, list]:
    """Resolve every config up front: a typo must fail now, not 8 GPU-hours in."""
    items, errors, seen = [], [], {}
    for c in configs:
        try:
            path = resolve_config_path(c)
            raw = B.load_config(path)
            eff = dispatch.resolve(path, overrides)  # raises on invalid / unrunnable configs
            if eff is None:  # runner without a resolver: identity from the raw config
                name = raw.get("name")
                epochs = (overrides or {}).get("epochs", raw.get("epochs", 0))
            else:
                name, epochs = eff["name"], eff["epochs"]
            if not isinstance(name, str) or not name:
                raise B.ConfigError("config has no `name`")
            if name in seen:
                raise B.ConfigError(f"run name {name!r} appears twice ({seen[name]} and {c})")
            seen[name] = c
            items.append(Item(
                config=path, name=name, family=B.family_of(name), arch=B.arch_of(name),
                epochs=int(epochs or 0),
                config_sha=B.config_sha256(eff) if eff is not None else None,
            ))
        except Exception as e:
            errors.append(f"{c}: {type(e).__name__}: {e}")
    return items, errors


# -------------------------------------------------------------- subprocess


def _tail_text(buf: bytes) -> str:
    text = buf.decode("utf-8", errors="replace").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[-TAIL_LINES:])[-ERROR_TAIL_CHARS:]


def _say(msg: str) -> None:
    """print() that never raises: a dead notebook output must not stop a run."""
    try:
        print(msg, flush=True)
    except (OSError, ValueError):
        pass


def _open_log(path: Path):
    return open(path, "ab")


class DriveLog:
    """logs/<run>.log on Drive, best effort. The first OSError (mount dropped,
    quota, EIO) closes it with ONE warning; the run it records goes on."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = None
        self.error: Optional[str] = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = _open_log(self.path)
        except OSError as e:
            self._give_up(e)

    def _give_up(self, e: BaseException) -> None:
        f, self.f = self.f, None
        self.error = f"{type(e).__name__}: {e}"
        _say(f"[matrix] WARNING: cannot write {self.path} ({self.error}) — the run continues; "
             "from here on its output is only in this cell")
        if f is not None:
            try:
                f.close()
            except (OSError, ValueError):
                pass

    def write(self, data: bytes) -> None:
        if self.f is not None:
            try:
                self.f.write(data)
            except (OSError, ValueError) as e:
                self._give_up(e)

    def flush(self) -> None:
        if self.f is not None:
            try:
                self.f.flush()
            except (OSError, ValueError) as e:
                self._give_up(e)

    def close(self) -> None:
        f, self.f = self.f, None
        if f is not None:
            try:
                f.close()
            except (OSError, ValueError):
                pass


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def tee_subprocess(req: RunRequest) -> LaunchResult:
    """Run req.cmd, streaming its output to our stdout AND req.log_path.

    Neither copy may cost the run: a failing Drive log is dropped with one
    warning (DriveLog), a dead stdout just stops the echo, and the child keeps
    training. The child is stopped only when the driver itself is interrupted
    (Ctrl-C, SystemExit) or this loop has a bug — an orphaned trainer would
    race the retry on the same run_dir."""
    tail = bytearray()
    log = DriveLog(req.log_path)
    try:
        header = (f"\n===== {utc_now()} {req.item.name} attempt {req.attempt}: "
                  f"{' '.join(shlex.quote(str(c)) for c in req.cmd)}\n")
        log.write(header.encode("utf-8"))
        log.flush()
        proc = subprocess.Popen(req.cmd, cwd=str(req.cwd), env=req.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        out = getattr(sys.stdout, "buffer", None)
        echo = True
        last_flush = time.monotonic()
        try:
            fd = proc.stdout.fileno()
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                if echo:
                    try:
                        if out is not None:
                            out.write(chunk)
                            out.flush()
                        else:
                            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                            sys.stdout.flush()
                    except (OSError, ValueError):
                        echo = False  # the notebook output is gone; the log still gets everything
                log.write(chunk)
                if time.monotonic() - last_flush > 10:
                    log.flush()
                    last_flush = time.monotonic()
                tail.extend(chunk)
                if len(tail) > TAIL_BYTES:
                    del tail[: len(tail) - TAIL_BYTES]
            rc = proc.wait()
        except BaseException:  # Ctrl-C / SystemExit, or a bug here (Drive + stdout errors never get here)
            _stop(proc)
            raise
    finally:
        log.close()
    return LaunchResult(rc, _tail_text(bytes(tail)))


# ---------------------------------------------------------------- the log


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(B.to_jsonable(record), ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def read_jsonl(path: Path) -> list:
    try:
        if not Path(path).is_file():
            return []
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:  # only seeds the ETA: never worth stopping the matrix for
        _say(f"[matrix] WARNING: cannot read {path} ({type(e).__name__}: {e}); ETA starts unmeasured")
        return []
    out = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


class RateBook:
    """Measured GPU minutes per epoch, per family and per architecture."""

    def __init__(self):
        self.family: dict = {}
        self.arch: dict = {}

    def add(self, family: str, arch: str, minutes: float, epochs: Optional[int]) -> None:
        if not epochs or epochs <= 0 or not minutes or minutes <= 0:
            return
        for table, key in ((self.family, family), (self.arch, arch)):
            m, e = table.get(key, (0.0, 0))
            table[key] = (m + minutes, e + epochs)

    @classmethod
    def from_log(cls, records: list) -> "RateBook":
        book = cls()
        for r in records:
            if r.get("status") == "ok":
                name = r.get("run_name") or ""
                book.add(B.family_of(name), B.arch_of(name), r.get("minutes") or 0.0,
                         r.get("epochs_this_attempt"))
        return book

    def rate(self, family: str, arch: str) -> tuple[Optional[float], str]:
        if family in self.family:
            m, e = self.family[family]
            return m / e, "measured"
        if arch in self.arch:
            m, e = self.arch[arch]
            return m / e, f"from {arch}"
        return None, "unmeasured"


def eta_line(items: list, book: RateBook) -> str:
    if not items:
        return "ETA: nothing left to run"
    total, parts, unknown = 0.0, [], []
    for it in items:
        rate, basis = book.rate(it.family, it.arch)
        if rate is None:
            unknown.append(it.name)
            continue
        total += rate * it.epochs
        parts.append(f"{it.family} {rate:.2f} min/epoch ({basis})")
    if not parts:
        return (f"ETA: unknown until a run of the same architecture finishes "
                f"({len(items)} run(s) remaining; L4 guide: ~50-90 min per 120-epoch run)")
    msg = f"ETA: ~{total / 60:.1f} h for {len(items) - len(unknown)} of {len(items)} remaining run(s)"
    msg += " [" + "; ".join(sorted(set(parts))) + "]"
    if unknown:
        msg += f"; no measurement yet for {len(unknown)} run(s) ({', '.join(unknown[:4])}"
        msg += ", ...)" if len(unknown) > 4 else ")"
    return msg


# ----------------------------------------------------------------- the run


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=ML_DIR, timeout=30).stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _bundle_from_tail(tail: str) -> Optional[Path]:
    for line in reversed(tail.splitlines()):
        if line.startswith(dispatch.BUNDLE_MARKER):
            return Path(line[len(dispatch.BUNDLE_MARKER):].strip())
    return None


def _run_state(run_root: Path, mirror_dir: Path, name: str) -> Optional[dict]:
    """The runner's attempt log (zander_run.json), local first, then the mirror."""
    for d in (run_root / name, mirror_dir / name):
        try:
            doc = json.loads((d / "zander_run.json").read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return doc
        except (OSError, ValueError):
            continue
    return None


def _attempt_facts(state: Optional[dict], started: str, resumed_hint: bool):
    """(epochs_completed, epochs trained by THIS attempt, resumed) from the
    runner's attempt log; the attempt counts only if it began after `started`."""
    attempts = (state or {}).get("attempts") or []
    ends = [a.get("end_epoch") or 0 for a in attempts]
    epochs_completed = max(ends) if ends else None
    last = attempts[-1] if attempts else None
    if last is None or str(last.get("started") or "") < started:
        return epochs_completed, None, resumed_hint
    trained = (last.get("end_epoch") or 0) - (last.get("start_epoch") or 0)
    return epochs_completed, trained, bool(last.get("resumed"))


def publish_bundle(local_zip: Path, bundles_dir: Path) -> str:
    """Copy a verified local bundle to Drive: zip first, verify, then the marker."""
    check = B.verify_bundle(local_zip)
    if not check.ok:
        raise RuntimeError(f"local bundle {local_zip} invalid: {check.reason}")
    dest = bundles_dir / local_zip.name
    B.atomic_copy(local_zip, dest)
    copied = B.sha256_file(dest)
    if copied != check.sha256:
        raise RuntimeError(f"bundle copy corrupted ({copied[:12]} != {check.sha256[:12]})")
    B.atomic_write_text(B.sha_path_for(dest), f"{check.sha256}  {dest.name}\n")
    return check.sha256


def _move_invalid(bundles_dir: Path, name: str, reason: str) -> None:
    zip_path = bundles_dir / f"{name}_bundle.zip"
    stamp = _stamp()
    for p in (zip_path, B.sha_path_for(zip_path)):
        if p.exists():
            target = p.with_name(f"{p.name}.invalid-{stamp}")
            shutil.move(str(p), str(target))
            print(f"[matrix] {name}: moved an invalid bundle file aside ({reason}): {target.name}", flush=True)


def _is_file(path: Path) -> bool:
    """Path.is_file() that answers False instead of raising on a dropped Drive
    mount (ENOTCONN / EIO are not swallowed by pathlib)."""
    try:
        return Path(path).is_file()
    except OSError:
        return False


class Matrix:
    def __init__(self, args, items: list, launcher: Launcher, run_root: Path):
        self.args = args
        self.items = items
        self.launcher = launcher
        self.run_root = run_root
        self.drive = Path(args.drive)
        self.bundles = self.drive / "bundles"
        self.mirror = self.drive / "mirror"
        self.logs = self.drive / "logs"
        self.jsonl = self.drive / "matrix_log.jsonl"
        for d in (self.bundles, self.mirror, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self.gpu = detect_gpu() or "cpu"
        self.profile = profile_for(args.profile, None if self.gpu == "cpu" else self.gpu)
        self.knobs = runtime_knobs(self.profile, os.cpu_count())
        self.git_sha = _git_sha()
        self.book = RateBook.from_log(read_jsonl(self.jsonl))
        self.jsonl_error: Optional[str] = None

    def record(self, item: Item, **fields) -> dict:
        rec = {
            "run_name": item.name, "config": str(item.config), "status": None, "attempt": 0,
            "resumed": None, "started": None, "ended": None, "minutes": 0.0,
            "epochs_completed": None, "gpu": self.gpu, "git_sha": self.git_sha,
            "returncode": None, "bundle_sha256": None, "error_tail": None,
            "epochs_this_attempt": None, "config_sha256": item.config_sha,
            "profile": self.profile, "workers": self.knobs.get("workers"),
            "cache": self.knobs.get("cache"), "overrides": self.args.overrides,
            "host": platform.node(), "warning": None,
        }
        rec.update(fields)
        try:
            append_jsonl(self.jsonl, rec)
            self.jsonl_error = None
        except (OSError, ValueError) as e:  # the log is bookkeeping: never stop the matrix for it
            if self.jsonl_error is None:
                _say(f"[matrix] WARNING: cannot append to {self.jsonl} ({type(e).__name__}: {e}) — "
                     "the matrix continues; the bundles on Drive stay the record of what finished")
            self.jsonl_error = f"{type(e).__name__}: {e}"
        return rec

    def command(self, item: Item) -> list:
        cmd = [sys.executable, "-u", str(ML_DIR / "train" / "dispatch.py"),
               "--config", str(item.config), "--mirror-dir", str(self.mirror)]
        if self.args.device is not None:
            cmd += ["--device", str(self.args.device)]
        for s in self.args.overrides:
            cmd += ["--set", s]
        if "workers" in self.knobs:
            cmd += ["--workers", str(self.knobs["workers"])]
        if "cache" in self.knobs:
            cmd += ["--cache", str(self.knobs["cache"])]
        return cmd

    def _unreadable(self, item: Item, why: str) -> Outcome:
        """Drive could not be READ: the bundle is unknown, not invalid."""
        note = f"{why} — NOT retrained, nothing moved (the bundle may be fine); rerun once Drive is readable"
        _say(f"[matrix] WARNING {item.name}: {note}")
        now = utc_now()
        rec = self.record(item, status="failed", started=now, ended=now, error_tail=note)
        return Outcome(item.name, "failed", note=f"Drive unreadable: {why}"[:120], records=[rec])

    def skip_if_done(self, item: Item) -> Optional[Outcome]:
        zip_path = self.bundles / f"{item.name}_bundle.zip"
        check = B.verify_bundle(zip_path)
        if check.io_error:
            return self._unreadable(item, f"checking {zip_path.name}: {check.reason}")
        if check.ok:
            meta = check.meta or {}
            warnings = []
            have = meta.get("config_sha256")
            if item.config_sha and have and have != item.config_sha:
                warnings.append(f"existing bundle was trained with config_sha256 {have[:12]} but today's "
                                f"effective config is {item.config_sha[:12]} — kept, NOT retrained "
                                "(delete the bundle to retrain)")
            for w in warnings:
                _say(f"[matrix] WARNING {item.name}: {w}")
            warning = "; ".join(warnings) or None
            leftover = self.mirror / item.name
            try:
                if leftover.exists():
                    shutil.rmtree(leftover, ignore_errors=True)
            except OSError:
                pass
            now = utc_now()
            rec = self.record(item, status="skipped", started=now, ended=now,
                              epochs_completed=meta.get("epochs_completed"),
                              bundle_sha256=check.sha256, warning=warning)
            _say(f"[matrix] {item.name}: skipped (bundle exists, sha {check.sha256[:12]})")
            return Outcome(item.name, "skipped", epochs_completed=meta.get("epochs_completed"),
                           note=warning or "bundle exists", records=[rec])
        try:
            if zip_path.exists() or B.sha_path_for(zip_path).exists():
                _move_invalid(self.bundles, item.name, check.reason)
        except OSError as e:  # retraining over a file we could not move aside would overwrite it
            return self._unreadable(item, f"moving the invalid {zip_path.name} aside: {type(e).__name__}: {e}")
        return None

    def run_item(self, item: Item) -> Outcome:
        done = self.skip_if_done(item)
        if done is not None:
            return done
        outcome = Outcome(item.name, "failed")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        for attempt in range(1, self.args.max_attempts + 1):
            resumed_hint = (_is_file(self.mirror / item.name / "last.pt")
                            or _is_file(self.run_root / item.name / "weights" / "last.pt"))
            started = utc_now()
            outcome.records.append(self.record(item, status="started", attempt=attempt,
                                               resumed=resumed_hint, started=started))
            print(f"[matrix] {item.name}: attempt {attempt}/{self.args.max_attempts} "
                  f"({'resume' if resumed_hint else 'fresh'})", flush=True)
            req = RunRequest(item, attempt, self.command(item), env, ML_DIR,
                             self.logs / f"{item.name}.log", self.mirror, self.run_root)
            t0 = time.monotonic()
            try:
                result = self.launcher(req)
            except KeyboardInterrupt:
                self.record(item, status="interrupted", attempt=attempt, resumed=resumed_hint,
                            started=started, ended=utc_now(),
                            minutes=round((time.monotonic() - t0) / 60, 2))
                raise
            except Exception as e:  # the driver failed around the run: a failed attempt, never a dead loop
                result = LaunchResult(None, f"matrix driver error: {type(e).__name__}: {e}")
            minutes = round((time.monotonic() - t0) / 60, 2)
            epochs_completed, this_attempt, resumed = _attempt_facts(
                _run_state(self.run_root, self.mirror, item.name), started, resumed_hint)
            outcome.tries, outcome.minutes = attempt, round(outcome.minutes + minutes, 2)
            outcome.resumed = resumed
            common = dict(attempt=attempt, resumed=resumed, started=started, minutes=minutes,
                          returncode=result.returncode, epochs_this_attempt=this_attempt,
                          epochs_completed=epochs_completed)
            error = None
            if result.returncode == 0:
                local_zip = _bundle_from_tail(result.tail) or (self.run_root / f"{item.name}_bundle.zip")
                try:
                    sha = publish_bundle(local_zip, self.bundles)
                    meta = B.read_bundle_meta(local_zip)
                    common["epochs_completed"] = meta.get("epochs_completed", epochs_completed)
                    shutil.rmtree(self.mirror / item.name, ignore_errors=True)
                    rec = self.record(item, status="ok", ended=utc_now(), bundle_sha256=sha, **common)
                    outcome.records.append(rec)
                    outcome.status, outcome.epochs_completed = "ok", common["epochs_completed"]
                    outcome.note = f"bundle {sha[:12]}"
                    self.book.add(item.family, item.arch, minutes, this_attempt)
                    _say(f"[matrix] {item.name}: OK in {minutes:.1f} min -> {self.bundles / local_zip.name}")
                    return outcome
                except Exception as e:
                    error = f"publishing the bundle failed: {type(e).__name__}: {e}"
            else:
                error = result.tail or f"exit code {result.returncode}"
            outcome.records.append(self.record(item, status="failed", ended=utc_now(),
                                               error_tail=error[-ERROR_TAIL_CHARS:], **common))
            outcome.epochs_completed = epochs_completed
            outcome.note = (error.strip().splitlines() or ["?"])[-1][:120]
            exit_txt = "driver error" if result.returncode is None else f"exit {result.returncode}"
            _say(f"[matrix] {item.name}: FAILED ({exit_txt}, attempt {attempt})")
            if result.returncode == dispatch.EXIT_CONFIG:
                outcome.note = "config/data error (not retried): " + outcome.note
                break
        return outcome


def summary_table(outcomes: list) -> str:
    head = f"{'run':<28} {'status':<8} {'tries':>5} {'resumed':>8} {'minutes':>8} {'epochs':>6}  note"
    rows = [head, "-" * len(head)]
    for o in outcomes:
        resumed = "-" if o.resumed is None else ("yes" if o.resumed else "no")
        epochs = "-" if o.epochs_completed is None else str(o.epochs_completed)
        rows.append(f"{o.name:<28} {o.status:<8} {o.tries:>5} {resumed:>8} {o.minutes:>8.1f} "
                     f"{epochs:>6}  {o.note}")
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train/matrix.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True, help="run configs, in priority order")
    p.add_argument("--drive", required=True, help="Drive folder (MyDrive/ZanderML) or any local dir")
    p.add_argument("--device", default=None, help="passed to dispatch.py (e.g. 0, cpu)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override for EVERY run (recorded in each bundle)")
    p.add_argument("--profile", default="auto", choices=PROFILE_CHOICES,
                   help="GPU profile for runtime knobs (workers / disk cache)")
    p.add_argument("--max-attempts", type=int, default=2, help="tries per run in this session")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    return p


def main(argv: Optional[list] = None, *, launcher: Optional[Launcher] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_attempts < 1:
        print("[matrix] --max-attempts must be >= 1", file=sys.stderr)
        return 2
    try:
        overrides = B.parse_overrides(args.overrides)
    except B.ConfigError as e:
        print(f"[matrix] {e}", file=sys.stderr)
        return 2
    items, errors = load_items(args.configs, overrides)
    if errors:
        print("[matrix] configuration errors — nothing was run:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2
    from config.settings import settings

    matrix = Matrix(args, items, launcher or tee_subprocess, settings.run_dirs_dir)
    print(f"[matrix] {len(items)} config(s); drive {matrix.drive}; gpu {matrix.gpu}; "
          f"profile {matrix.profile or 'none'} {matrix.knobs}; git {matrix.git_sha[:12]}", flush=True)
    # cheap existence check for the plan/ETA; run_item re-verifies the sha before skipping
    pending = [it for it in items if not (
        _is_file(matrix.bundles / f"{it.name}_bundle.zip")
        and _is_file(B.sha_path_for(matrix.bundles / f"{it.name}_bundle.zip")))]
    print(f"[matrix] {len(items) - len(pending)} already bundled; {eta_line(pending, matrix.book)}",
          flush=True)
    if args.dry_run:
        for it in items:
            state = ("done" if it not in pending else
                     "resume" if _is_file(matrix.mirror / it.name / "last.pt") else "fresh")
            print(f"  {it.name:<28} {state:<7} {it.epochs} epochs  {it.config}", flush=True)
        return 0
    outcomes = []
    try:
        for i, item in enumerate(items):
            try:
                outcomes.append(matrix.run_item(item))
            except Exception as e:  # last resort: one run's trouble never ends the session
                note = f"matrix driver error: {type(e).__name__}: {e}"
                _say(f"[matrix] {item.name}: FAILED ({note})")
                outcomes.append(Outcome(item.name, "failed", note=note[:120]))
            rest = [it for it in items[i + 1:] if it in pending]
            print(f"[matrix] {eta_line(rest, matrix.book)}", flush=True)
    except KeyboardInterrupt:
        print("\n[matrix] interrupted — finished bundles are safe; rerun to resume", flush=True)
        print(summary_table(outcomes), flush=True)
        return 130
    print("\n" + summary_table(outcomes), flush=True)
    failed = [o.name for o in outcomes if o.status == "failed"]
    if failed:
        print(f"[matrix] {len(failed)} run(s) failed: {', '.join(failed)} — rerun the same "
              "command to retry (partial runs resume from the mirror)", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
