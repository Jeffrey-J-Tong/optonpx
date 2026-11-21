"""
What notebooks 2.1 / 2.2 / 2.3 depend on, and whether their outputs in a processed session are
complete and still match the metadata.

Only the parameters that change a notebook's output are compared ("relevant"); every other
difference between the metadata copy stored by notebook 2.1 and the current metadata is only
reported ("other"). Used by notebook 2.1 (resume check) and by scripts/run_batch.py (skip check).

Nothing here writes or deletes files.
"""

import copy
import json
from datetime import datetime
from pathlib import Path

import yaml

from src.preprocess import DEFAULT_ARTIFACT_MASK, DEFAULT_PULSE_DETECTION

# defaults of the per-probe preprocess blocks of the metadata (notebook 2.1)
DEFAULT_PREPROCESS_BLOCKS = {
    "phase_shift": {"apply": True},                                   # + margin_ms / chunk_s from the metadata
    "mask_laser_artifact": {**DEFAULT_ARTIFACT_MASK, "apply": True},
    "car": {"apply": False},                                          # Kilosort does CAR
    "mask_noise_artifact": {"apply": True, "threshold_mad": 8.0, "min_fraction": 0.5,
                            "min_slope_mad": 4.0,                     # sharp events only
                            "pre_ms": 0.5, "post_ms": 1.0,
                            "max_optotagging_events": 10},
}

# kilosort_params keys used by notebooks 2.2 / 2.3 themselves; every other key is a Kilosort setting
KILOSORT_OWN_KEYS = {"bad_channels", "do_car", "output_dir"}

PREPROCESS_SIDECARS = ("session_boundaries.json", "preprocessing.json", "noise_events.csv")


def _json_normal(value):
    """The value as it reads back from JSON (tuples -> lists, numpy / paths -> str)."""
    return json.loads(json.dumps(value, default=str))


def nested_differences(old, new, name):
    """'name.key: old -> new' for every value that differs between two nested dicts."""
    if isinstance(old, dict) and isinstance(new, dict):
        differences = []
        for key in sorted(set(old) | set(new), key=str):
            if key not in old:
                differences.append(f"{name}.{key}: (missing) -> {new[key]!r}")
            elif key not in new:
                differences.append(f"{name}.{key}: {old[key]!r} -> (missing)")
            else:
                differences += nested_differences(old[key], new[key], f"{name}.{key}")
        return differences
    return [] if old == new else [f"{name}: {old!r} -> {new!r}"]


def _outside(differences, prefixes):
    """The differences whose key path does not start with any of `prefixes`."""
    return [d for d in differences
            if not any(d.startswith(p + ".") or d.startswith(p + ":") for p in prefixes)]


# --------------------------------------------------------------------------- #
# notebook 2.1
# --------------------------------------------------------------------------- #

def preprocess_parameters(config, spikeinterface_version):
    """Every parameter that changes the output of notebook 2.1: the metadata blocks merged with the
    defaults, the raw session folders that are read, the laser detection and the SpikeInterface
    version. As stored in <processed_session>/preprocess_checkpoints/parameters.json."""
    probes = {}
    for probe_name in config["streams"]["probes"]:
        block = config[probe_name]["preprocess"]
        order = list(block["concatenate"])
        probes[probe_name] = {
            "concatenate": order,
            "sessions": {role: config["sessions"][role] for role in order},
            **{name: {**defaults, **(block.get(name) or {})}
               for name, defaults in DEFAULT_PREPROCESS_BLOCKS.items()},
            "bad_channels": config[probe_name]["kilosort_params"]["bad_channels"]}
    return _json_normal({
        "spikeinterface": spikeinterface_version,
        "laser_channels": {name: channel for name, channel in config["adc"]["channels"].items()
                           if name.endswith("_laser")},
        "pulse_detection": {**DEFAULT_PULSE_DETECTION, "threshold_volts": config["adc"]["threshold_volts"]},
        "probes": probes})


def _preprocess_metadata_prefixes(config):
    """Metadata key paths that feed preprocess_parameters (their changes show up there)."""
    prefixes = ["metadata.adc", "metadata.streams"]
    for probe_name in config["streams"]["probes"]:
        prefixes += [f"metadata.{probe_name}.preprocess", f"metadata.{probe_name}.kilosort_params.bad_channels"]
        prefixes += [f"metadata.sessions.{role}" for role in config[probe_name]["preprocess"]["concatenate"]]
    return prefixes


def preprocess_differences(old_parameters, old_metadata, new_parameters, config):
    """(relevant, warning, other) differences between an earlier run of notebook 2.1 and now.

    relevant: changes of preprocess_parameters -> the earlier output no longer matches.
    warning:  changed bad_channels. 2.1 only leaves them out of the noise detection and the laser lag
              estimate, where one channel more or less among ~380 hardly matters, so the earlier output
              is kept; the channel map (2.2) and the sorting (2.3) must follow the new list.
    other:    the remaining changes of the metadata (old copy -> current), which do not change the
              output of notebook 2.1 (e.g. kilosort settings, target region, laser_protocol)."""
    old = copy.deepcopy(old_parameters)
    for probe in old.get("probes", {}).values():
        if "sessions" not in probe:   # records written before the session folders were stored
            probe["sessions"] = {role: (old_metadata or {}).get("sessions", {}).get(role)
                                 for role in probe.get("concatenate", [])}
    differences = nested_differences(old, new_parameters, "parameters")
    warning = [d for d in differences if d.split(":")[0].endswith(".bad_channels")]
    relevant = [d for d in differences if d not in warning]
    other = _outside(nested_differences(_json_normal(old_metadata or {}), _json_normal(config), "metadata"),
                     _preprocess_metadata_prefixes(config))
    return relevant, warning, other


def preprocess_status(processed_session, config, metadata_name, spikeinterface_version):
    """State of notebook 2.1's output in processed_session:
    "absent"  folder missing or empty;
    "done"    every probe written (.preprocess_done), sidecars present, no relevant change;
    "partial" started, not finished, no relevant change (2.1 resumes);
    "changed" relevant parameters differ from the earlier run;
    "unknown" the folder holds files but no record of an earlier 2.1 run.
    Returns {"state", "relevant", "warning", "other", "detail"} (see preprocess_differences)."""
    processed_session = Path(processed_session)
    result = {"state": "absent", "relevant": [], "warning": [], "other": [], "detail": ""}
    if not processed_session.exists() or not any(processed_session.iterdir()):
        return result
    parameters_path = processed_session / "preprocess_checkpoints" / "parameters.json"
    metadata_copy_path = processed_session / metadata_name
    if not (parameters_path.exists() and metadata_copy_path.exists()):
        result.update(state="unknown",
                      detail=f"no {metadata_name} / preprocess_checkpoints/parameters.json in a non-empty folder")
        return result
    relevant, warning, other = preprocess_differences(
        json.loads(parameters_path.read_text(encoding="utf-8")),
        yaml.safe_load(metadata_copy_path.read_text(encoding="utf-8")),
        preprocess_parameters(config, spikeinterface_version), config)
    result.update(relevant=relevant, warning=warning, other=other)
    if relevant:
        result.update(state="changed", detail=f"{len(relevant)} relevant parameter(s) changed")
        return result
    missing = [f"{probe_name}/.preprocess_done" for probe_name in config["streams"]["probes"]
               if not (processed_session / probe_name / ".preprocess_done").exists()]
    missing += [name for name in PREPROCESS_SIDECARS if not (processed_session / name).exists()]
    result.update(state="partial" if missing else "done",
                  detail=f"missing {missing}" if missing else "")
    return result


# --------------------------------------------------------------------------- #
# notebook 2.2
# --------------------------------------------------------------------------- #

def kilosort_results_dir(processed_session, config, probe_name):
    return Path(processed_session) / probe_name / config[probe_name]["kilosort_params"].get("output_dir", "kilosort4")


def _excluded_channels(mat_path, n_chan):
    import scipy.io
    kept = set((scipy.io.loadmat(mat_path)["chanMap"].ravel() - 1).astype(int).tolist())
    return sorted(set(range(n_chan)) - kept)


def channel_map_status(processed_session, config, session_name):
    """State of notebook 2.2's channel maps: "done" (every probe has one .mat that excludes exactly
    the metadata bad_channels), "todo" (missing or different, no Kilosort results yet: 2.2 rewrites
    it), "changed" (different, and Kilosort results made with the old map exist).
    Needs notebook 2.1's session_boundaries.json. Returns {"state", "relevant", "detail"}."""
    processed_session = Path(processed_session)
    streams = json.loads((processed_session / "session_boundaries.json").read_text())["streams"]
    todo, changed = [], []
    for probe_name in config["streams"]["probes"]:
        bad_channels = sorted(config[probe_name]["kilosort_params"]["bad_channels"])
        mat_paths = sorted((processed_session / probe_name).glob(f"{session_name}_*.mat"))
        if len(mat_paths) == 1:
            excluded = _excluded_channels(mat_paths[0], int(streams[probe_name]["n_chan"]))
            if excluded == bad_channels:
                continue
            reason = f"{probe_name}: channel map excludes {excluded}, metadata bad_channels {bad_channels}"
        else:
            reason = f"{probe_name}: {len(mat_paths)} channel map(s)"
        results_dir = kilosort_results_dir(processed_session, config, probe_name)
        (changed if results_dir.exists() and any(results_dir.iterdir()) else todo).append(reason)
    if changed:
        return {"state": "changed", "relevant": changed + todo, "detail": "Kilosort results use another channel map"}
    return {"state": "todo" if todo else "done", "relevant": todo, "detail": "; ".join(todo)}


# --------------------------------------------------------------------------- #
# notebook 2.3
# --------------------------------------------------------------------------- #

def kilosort_status(processed_session, config, kilosort_version=None):
    """State of notebook 2.3's results per probe, from kilosort_run.json:
    "done"       every probe sorted with the current settings, bad channels, data and Kilosort version;
    "todo"       some probes have no results folder yet (2.3 sorts those, skips the others);
    "changed"    a probe was sorted with other settings / channels / data, or before 2.1 rewrote the data;
    "unfinished" a results folder without kilosort_run.json (crashed run, or made outside notebook 2.3).
    Returns {"state", "relevant", "detail"}."""
    processed_session = Path(processed_session)
    todo, changed, unfinished = [], [], []
    for probe_name in config["streams"]["probes"]:
        results_dir = kilosort_results_dir(processed_session, config, probe_name)
        run_path = results_dir / "kilosort_run.json"
        if not run_path.exists():
            if results_dir.exists() and any(results_dir.iterdir()):
                unfinished.append(f"{probe_name}: {results_dir} has no kilosort_run.json")
            else:
                todo.append(probe_name)
            continue
        run = json.loads(run_path.read_text())
        params = config[probe_name]["kilosort_params"]
        overrides = _json_normal({k: v for k, v in params.items() if k not in KILOSORT_OWN_KEYS})
        dat_path = processed_session / probe_name / "continuous.dat"
        differences = nested_differences(run.get("metadata_overrides", {}), overrides, f"{probe_name}.kilosort_params")
        if bool(run.get("do_CAR")) != bool(params["do_car"]):
            differences.append(f"{probe_name}.do_car: {run.get('do_CAR')} -> {params['do_car']}")
        excluded = run.get("channel_map", {}).get("excluded_channels")
        if excluded != sorted(params["bad_channels"]):
            differences.append(f"{probe_name}.bad_channels: {excluded} -> {sorted(params['bad_channels'])}")
        if not dat_path.exists() or run.get("data", {}).get("bytes") != dat_path.stat().st_size:
            differences.append(f"{probe_name}: continuous.dat is not the one that was sorted")
        done_marker = processed_session / probe_name / ".preprocess_done"
        if done_marker.exists() and run.get("started"):
            written = json.loads(done_marker.read_text()).get("finished")
            if written and datetime.fromisoformat(written) > datetime.fromisoformat(run["started"]):
                differences.append(f"{probe_name}: continuous.dat was rewritten ({written}) after sorting "
                                   f"({run['started']})")
        if kilosort_version and run.get("kilosort_version") != kilosort_version:
            differences.append(f"{probe_name}.kilosort_version: {run.get('kilosort_version')} -> {kilosort_version}")
        changed += differences
    if changed or unfinished:
        return {"state": "changed" if changed else "unfinished", "relevant": changed + unfinished,
                "detail": "; ".join(changed + unfinished)}
    return {"state": "todo" if todo else "done", "relevant": [],
            "detail": f"to sort: {todo}" if todo else ""}
