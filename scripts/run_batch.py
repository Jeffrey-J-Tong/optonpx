"""
Run notebooks 2.1 (preprocess) -> 2.2 (channel map) -> 2.3 (Kilosort) over several sessions,
unattended, with papermill.

    conda activate optonpx
    python scripts/run_batch.py --dry-run     # checks + plan only, runs nothing
    python scripts/run_batch.py

Set the sessions and drives in the settings block below.

- Raw data: for each session, the first drive of RAW_DRIVES that holds every raw session folder it
  needs (<drive>/<animal>/ephys_raw/<folder>) with complete streams (continuous.dat size matches
  timestamps.npy and the channel count). Backups on later drives are only used when an earlier one
  is missing or incomplete.
- Output: a session already processed on one of OUTPUT_DRIVES stays there. Otherwise the first drive
  of OUTPUT_DRIVES (avoiding the session's raw drive if possible) that keeps its reserve after
  SPACE_FACTOR x the raw data of the session. Space planned for earlier sessions of the batch counts.
- Skipping: a step whose output is complete and whose relevant parameters still match the metadata
  is skipped (src.pipeline_check). Changed relevant parameters are never overwritten: the session is
  marked for a manual decision and the batch goes on with the next session.
- Each step runs in its own kernel: 2.1 / 2.2 in the optonpx kernel, 2.3 in the kilosort kernel.
  A failed step stops that session; the batch goes on with the next one.
- logs/batch_<time>/: the executed notebooks (outputs, figures, errors), batch.log and summary.csv.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ============================== settings ==============================
SESSIONS = [                        # metadata/<name>.yaml
    "IM-1994_2026-11-10_DMS_behavior",
]
RAW_DRIVES = ["Z:/", "D:/", "E:/"]  # searched in this order (backups with the same content)
OUTPUT_DRIVES = ["F:/", "D:/"]      # first one with enough space is used
STEPS = ["2.1", "2.2", "2.3"]       # notebooks to run, in this order
PROCESSED_SUFFIX = ""               # as processed_suffix in the notebooks

SPACE_FACTOR = 1.5                  # needed space = factor x raw continuous.dat of the session
RESERVE_GB = 100                    # free space kept on every output drive: max(RESERVE_GB,
RESERVE_FRACTION = 0.05             #   RESERVE_FRACTION x capacity) ...
SYSTEM_DRIVE = "C:"                 # ... and at least SYSTEM_RESERVE_GB on the system drive
SYSTEM_RESERVE_GB = 200

KERNELS = {"2.1": "python3", "2.2": "python3", "2.3": "kilosort"}   # Jupyter kernel per notebook
# ======================================================================

NOTEBOOKS = {"2.1": "2.1_data_preprocessing.ipynb",
             "2.2": "2.2_generate_channel_map.ipynb",
             "2.3": "2.3_run_kilosort.ipynb"}
GB = 1e9


class Log:
    """print() to the console and to batch.log."""

    def __init__(self, path):
        self.path = path

    def __call__(self, *lines):
        text = "\n".join(str(line) for line in lines)
        print(text, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text + "\n")


# --------------------------------------------------------------------------- #
# checks before the run
# --------------------------------------------------------------------------- #

def drive_of(path):
    return Path(path).drive.upper()


def reserve_bytes(drive):
    total = shutil.disk_usage(drive).total
    reserve = max(RESERVE_GB * GB, RESERVE_FRACTION * total)
    if drive_of(drive) == SYSTEM_DRIVE.upper():
        reserve = max(reserve, SYSTEM_RESERVE_GB * GB)
    return reserve


def session_roles(config):
    """Raw sessions read for each probe (preprocess.concatenate)."""
    return {probe: list(config[probe]["preprocess"]["concatenate"]) for probe in config["streams"]["probes"]}


def check_raw_folder(folder, probes):
    """(bytes of continuous.dat per probe, problem or None) for one raw Open Ephys folder."""
    from src import oe_parse_folders
    try:
        _, oe_paths = oe_parse_folders(folder)
        oebin = json.loads(Path(oe_paths["oebin"]).read_text())
    except Exception as error:   # noqa: BLE001 -- any parse problem makes this copy unusable
        return {}, f"{folder}: {error}"
    channels = {entry["folder_name"].strip("/"): int(entry["num_channels"]) for entry in oebin["continuous"]}
    sizes = {}
    for probe in probes:
        streams = [s for s in oe_paths["ephys_streams"] if s.name.split(".")[-1].lower() == probe.lower()]
        if len(streams) != 1:
            return {}, f"{folder}: {len(streams)} streams for {probe}"
        dat, timestamps = streams[0] / "continuous.dat", streams[0] / "timestamps.npy"
        if not (dat.exists() and timestamps.exists()):
            return {}, f"{streams[0]}: continuous.dat / timestamps.npy missing"
        expected = len(np.load(timestamps, mmap_mode="r")) * channels[streams[0].name] * 2
        if dat.stat().st_size != expected:
            return {}, f"{dat}: {dat.stat().st_size} bytes, {expected} expected (incomplete copy?)"
        sizes[probe] = dat.stat().st_size
    return sizes, None


def find_raw(config, log):
    """(raw drive, total raw bytes read by the session) or (None, problem)."""
    roles = session_roles(config)
    needed = {}
    for probe, order in roles.items():
        for role in order:
            needed.setdefault(role, []).append(probe)
    notes = []
    for drive in RAW_DRIVES:
        if not Path(drive).exists():
            continue
        folders = {role: Path(drive) / config["animal"] / "ephys_raw" / config["sessions"][role] for role in needed}
        if not all(folder.exists() for folder in folders.values()):
            continue
        total, problem = 0, None
        for role, folder in folders.items():
            sizes, problem = check_raw_folder(folder, needed[role])
            if problem:
                break
            total += sum(sizes.values())
        if problem:
            notes.append(problem)
            continue
        for note in notes:
            log(f"  note: skipped an incomplete raw copy: {note}")
        return drive, total
    return None, ("raw folders " + str({role: config["sessions"][role] for role in needed})
                  + f" not found complete on {RAW_DRIVES}" + (f" ({'; '.join(notes)})" if notes else ""))


def processed_folder(drive, config):
    return Path(drive) / config["animal"] / "ephys" / (config["sessions"]["processed"] + PROCESSED_SUFFIX)


def plan_sessions(log):
    """One plan per session: config, raw / output drive, or the problem that stops it."""
    plans, planned_bytes, seen_folders = [], {}, {}
    for name in SESSIONS:
        plan = {"name": name, "problem": None}
        plans.append(plan)
        log(f"\n--- {name}")
        try:
            plan["config"] = config = yaml.safe_load((REPO / "metadata" / f"{name}.yaml").read_text(encoding="utf-8"))
            for probe, order in session_roles(config).items():
                unknown = [role for role in order if role == "processed" or role not in config["sessions"]]
                if unknown:
                    raise ValueError(f"{probe}.preprocess.concatenate lists {unknown}, not raw sessions")
        except Exception as error:   # noqa: BLE001
            plan["problem"] = f"metadata: {error}"
            log(f"  PROBLEM {plan['problem']}")
            continue
        if name in [p["name"] for p in plans[:-1]]:
            plan["problem"] = "listed twice"
            log(f"  PROBLEM {plan['problem']}")
            continue

        raw_drive, raw_bytes = None, 0
        if "2.1" in STEPS or "2.2" in STEPS:
            raw_drive, result = find_raw(config, log)
            if raw_drive is None:
                plan["problem"] = result
                log(f"  PROBLEM {result}")
                continue
            raw_bytes = result
            log(f"  raw     {raw_drive}  ({raw_bytes / GB:.0f} GB of continuous.dat to read)")
        plan["raw_drive"] = raw_drive

        existing = [d for d in OUTPUT_DRIVES if Path(d).exists() and processed_folder(d, config).exists()
                    and any(processed_folder(d, config).iterdir())]
        if len(existing) > 1:
            plan["problem"] = f"processed folder exists on several drives: {existing}"
            log(f"  PROBLEM {plan['problem']}")
            continue
        written = 0
        if existing:
            written = sum(f.stat().st_size for f in processed_folder(existing[0], config).glob("*/continuous.dat"))
        need = max(SPACE_FACTOR * raw_bytes - written, 0)

        def fits(drive):
            free = shutil.disk_usage(drive).free - planned_bytes.get(drive_of(drive), 0)
            return free - need >= reserve_bytes(drive), free

        if existing:
            out_drive = existing[0]
            ok, free = fits(out_drive)
            if not ok:
                plan["problem"] = (f"already started on {out_drive}, but {need / GB:.0f} GB more do not fit "
                                   f"(free {free / GB:.0f} GB, reserve {reserve_bytes(out_drive) / GB:.0f} GB)")
                log(f"  PROBLEM {plan['problem']}")
                continue
            log(f"  output  {processed_folder(out_drive, config)}  (existing; ~{need / GB:.0f} GB more)")
        else:
            candidates = [d for d in OUTPUT_DRIVES if Path(d).exists()]
            candidates = ([d for d in candidates if drive_of(d) != drive_of(raw_drive or "")]
                          + [d for d in candidates if drive_of(d) == drive_of(raw_drive or "")])
            out_drive = next((d for d in candidates if fits(d)[0]), None)
            if out_drive is None:
                plan["problem"] = (f"no output drive with {need / GB:.0f} GB + reserve free: "
                                   + ", ".join(f"{d} {fits(d)[1] / GB:.0f} GB free" for d in candidates))
                log(f"  PROBLEM {plan['problem']}")
                continue
            same = drive_of(out_drive) == drive_of(raw_drive or "")
            log(f"  output  {processed_folder(out_drive, config)}  (new; ~{need / GB:.0f} GB)"
                + ("  WARNING: same drive as the raw data (no other drive has space)" if same else ""))
        folder = str(processed_folder(out_drive, config)).lower()
        if folder in seen_folders:
            plan["problem"] = f"same processed folder as {seen_folders[folder]}"
            log(f"  PROBLEM {plan['problem']}")
            continue
        seen_folders[folder] = name
        planned_bytes[drive_of(out_drive)] = planned_bytes.get(drive_of(out_drive), 0) + need
        plan["out_drive"] = out_drive
        plan["processed"] = processed_folder(out_drive, config)
    return plans


def check_kernels(log):
    """Kernels exist; the kilosort kernel imports kilosort and sees a GPU. Returns the Kilosort version."""
    from jupyter_client.kernelspec import KernelSpecManager
    specs = KernelSpecManager().find_kernel_specs()
    missing = sorted({KERNELS[step] for step in STEPS} - set(specs))
    if missing:
        raise RuntimeError(f"Jupyter kernel(s) {missing} not found (available: {sorted(specs)}). Register the "
                           "kilosort env once: conda activate kilosort && python -m ipykernel install --user "
                           "--name kilosort --display-name \"Python (kilosort)\"")
    if "2.3" not in STEPS:
        return None
    python = KernelSpecManager().get_kernel_spec(KERNELS["2.3"]).argv[0]
    probe = ("import json, kilosort, torch; ok = torch.cuda.is_available(); "
             "free = torch.cuda.mem_get_info()[0] if ok else 0; "
             "print(json.dumps({'kilosort': kilosort.__version__, 'cuda': ok, "
             "'gpu': torch.cuda.get_device_name(0) if ok else None, 'free_gb': free / 1e9}))")
    result = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"kernel {KERNELS['2.3']!r} ({python}) cannot import kilosort / torch:\n{result.stderr[-1500:]}")
    info = json.loads(result.stdout.strip().splitlines()[-1])
    log(f"kernel {KERNELS['2.3']}: Kilosort {info['kilosort']}, GPU {info['gpu']} ({info['free_gb']:.1f} GB free)")
    if not info["cuda"]:
        raise RuntimeError(f"kernel {KERNELS['2.3']!r}: torch sees no CUDA GPU")
    return info["kilosort"]


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def step_status(step, plan, spikeinterface_version, kilosort_version):
    from src.pipeline_check import channel_map_status, kilosort_status, preprocess_status
    config, folder = plan["config"], plan["processed"]
    if step == "2.1":
        status = preprocess_status(folder, config, f"{plan['name']}.yaml", spikeinterface_version)
        return {"done": "done", "absent": "run", "partial": "run"}.get(status["state"], "manual"), status
    if not (folder / "session_boundaries.json").exists():
        return "manual", {"state": "missing input", "detail": "notebook 2.1 output missing", "relevant": []}
    status = (channel_map_status(folder, config, plan["name"]) if step == "2.2"
              else kilosort_status(folder, config, kilosort_version))
    return {"done": "done", "todo": "run"}.get(status["state"], "manual"), status


def run_notebook(step, plan, log_dir):
    import papermill as pm
    output = log_dir / f"{plan['name']}_{step}.ipynb"
    parameters = {"session_name": plan["name"], "base_dir_output": str(plan["out_drive"]),
                  "processed_suffix": PROCESSED_SUFFIX}
    if step in ("2.1", "2.2"):
        parameters.update(base_dir_raw=str(plan["raw_drive"]), is_overwrite=False)
    pm.execute_notebook(REPO / "notebook" / NOTEBOOKS[step], output, parameters=parameters,
                        kernel_name=KERNELS[step], cwd=str(REPO / "notebook"), progress_bar=True)
    return output


def write_summary(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ["session", "step", "result", "detail", "minutes", "notebook",
                                    "raw_drive", "output"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="checks and plan only, run nothing")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = REPO / "logs" / f"batch_{stamp}{'_dry-run' if args.dry_run else ''}"
    log_dir.mkdir(parents=True)
    log = Log(log_dir / "batch.log")
    log(f"batch {stamp}: {len(SESSIONS)} session(s), steps {STEPS}", f"logs: {log_dir}")

    unknown_steps = [step for step in STEPS if step not in NOTEBOOKS]
    if unknown_steps:
        raise ValueError(f"unknown steps {unknown_steps}")
    import spikeinterface
    kilosort_version = check_kernels(log)
    plans = plan_sessions(log)

    rows = []
    for plan in plans:
        if plan["problem"]:
            rows.append({"session": plan["name"], "step": "-", "result": "not started", "detail": plan["problem"]})
    write_summary(rows, log_dir / "summary.csv")
    runnable = [plan for plan in plans if not plan["problem"]]
    log(f"\n{len(runnable)} of {len(plans)} session(s) can run")
    if args.dry_run:
        for plan in runnable:
            for step in STEPS:
                try:
                    action, status = step_status(step, plan, spikeinterface.__version__, kilosort_version)
                except Exception as error:   # noqa: BLE001
                    action, status = "manual", {"detail": str(error), "relevant": [], "other": []}
                log(f"  {plan['name']} {step}: {action}  {status.get('detail', '')}")
                for line in status.get("warning", []):
                    log(f"      WARNING (kept; 2.2 / 2.3 must follow it): {line}")
                for line in status.get("relevant", []):
                    log(f"      {line}")
                if action != "done":
                    break    # later steps depend on this one (run) or the session stops here (manual)
        log("dry run: nothing was run")
        return

    for plan in runnable:
        log(f"\n========== {plan['name']}  (raw {plan['raw_drive']}, output {plan['processed']}) ==========")
        for step in STEPS:
            row = {"session": plan["name"], "step": step, "raw_drive": plan["raw_drive"],
                   "output": str(plan["processed"]), "detail": "", "minutes": "", "notebook": ""}
            rows.append(row)
            try:
                action, status = step_status(step, plan, spikeinterface.__version__, kilosort_version)
            except Exception as error:   # noqa: BLE001
                action, status = "manual", {"detail": f"status check failed: {error}", "relevant": []}
            for line in status.get("warning", []):
                log(f"  {step} WARNING (kept; 2.2 / 2.3 must follow it): {line}")
            for line in status.get("other", []):
                log(f"  {step} info (no effect on {step}): {line}")
            if action == "done":
                row.update(result="skipped (done)")
                log(f"{step}: already done, skipped")
                continue
            if action == "manual":
                row.update(result="manual decision", detail=status.get("detail", "") + (
                    ": " + "; ".join(status["relevant"]) if status.get("relevant") else ""))
                log(f"{step}: NEEDS A MANUAL DECISION -- {row['detail']}", "  (the rest of this session is skipped)")
                break
            started = datetime.now()
            log(f"{step}: running ({started:%H:%M:%S}) -> {log_dir / (plan['name'] + '_' + step + '.ipynb')}")
            try:
                row["notebook"] = str(run_notebook(step, plan, log_dir))
                error = None
            except Exception as exception:   # noqa: BLE001 -- PapermillExecutionError, dead kernel, ...
                error = exception
                row["notebook"] = str(log_dir / f"{plan['name']}_{step}.ipynb")
            row["minutes"] = f"{(datetime.now() - started).total_seconds() / 60:.1f}"
            if error is not None:
                evalue = str(getattr(error, "evalue", error)).strip()
                message = f"{getattr(error, 'ename', type(error).__name__)}: {evalue.splitlines()[0] if evalue else ''}"
                if "noise_decisions" in evalue:
                    row.update(result="waiting (noise_decisions.yaml)", detail=message)
                    log(f"{step}: STOPPED -- fill in {plan['processed'] / 'noise_decisions.yaml'} and run the batch again")
                else:
                    row.update(result="failed", detail=message)
                    log(f"{step}: FAILED after {row['minutes']} min -- {message}")
                write_summary(rows, log_dir / "summary.csv")
                break
            action, status = step_status(step, plan, spikeinterface.__version__, kilosort_version)
            if action != "done":
                row.update(result="failed", detail=f"ran, but the output is not complete: {status.get('detail', '')}")
                log(f"{step}: FAILED -- {row['detail']}")
                write_summary(rows, log_dir / "summary.csv")
                break
            row.update(result="ran")
            log(f"{step}: done in {row['minutes']} min")
            write_summary(rows, log_dir / "summary.csv")
        write_summary(rows, log_dir / "summary.csv")

    log("\n========== summary ==========")
    for row in rows:
        log(f"{row['session']:40s} {row['step']:4s} {row['result']:32s} {row.get('detail', '')}")
    log(f"\n{log_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
