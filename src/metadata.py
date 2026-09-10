"""
Structured metadata for organising raw Open Ephys recordings and defining
concatenated ("merged") recordings.

- One *session* YAML per raw Open Ephys recording folder (behavior or
  optotagging), describing where it is, which probe streams to use, and -- for
  optotagging sessions -- the laser sync channels and artifact-masking params.
- One *merge recipe* YAML per concatenated output, listing the session YAMLs in
  concatenation order and the output location.

``notebook/2.2_concatenate_sessions.ipynb`` reads a merge recipe.
"""

from pathlib import Path

import yaml

from src.preprocess import (
    DEFAULT_PULSE_DETECTION,
    DEFAULT_LAG_ESTIMATION,
    DEFAULT_ARTIFACT_MASK,
)

SESSION_TYPES = ("behavior", "optotagging")

_REQUIRED_SESSION_KEYS = ("session_id", "session_type", "open_ephys", "probe_streams")
_REQUIRED_RECIPE_KEYS = ("merge_id", "output_dir", "probe_streams", "sessions")

_PHASE_SHIFT_DEFAULTS = {"apply": True, "margin_ms": 40.0, "chunk_s": 10.0}


def _read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a YAML mapping at the top level")
    return data


def load_session_metadata(path) -> dict:
    """
    Load and validate one session metadata YAML.

    Fills optotagging sub-blocks (``pulse_detection``, ``lag_estimation``,
    ``artifact_mask``) with the Berke-lab-pipeline defaults where keys are
    omitted. Adds ``_path`` (the resolved YAML path, as a string).
    """
    path = Path(path).resolve()
    meta = _read_yaml(path)

    missing = [k for k in _REQUIRED_SESSION_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{path.name}: missing required field(s) {missing}")

    if not isinstance(meta["open_ephys"], dict) or "folder_path" not in meta["open_ephys"]:
        raise ValueError(f"{path.name}: open_ephys.folder_path is required")

    if meta["session_type"] not in SESSION_TYPES:
        raise ValueError(
            f"{path.name}: session_type must be one of {SESSION_TYPES}, "
            f"got {meta['session_type']!r}"
        )

    if not isinstance(meta["probe_streams"], list) or not meta["probe_streams"]:
        raise ValueError(f"{path.name}: probe_streams must be a non-empty list")

    if meta["session_type"] == "optotagging":
        opto = meta.get("optotagging")
        if not isinstance(opto, dict) or not opto.get("laser_inputs"):
            raise ValueError(
                f"{path.name}: an optotagging session needs "
                f"optotagging.laser_inputs (list of {{color, adc_channel}})"
            )
        for li in opto["laser_inputs"]:
            if "color" not in li or "adc_channel" not in li:
                raise ValueError(
                    f"{path.name}: every optotagging.laser_inputs entry needs "
                    f"'color' and 'adc_channel'"
                )
        opto["pulse_detection"] = {**DEFAULT_PULSE_DETECTION, **opto.get("pulse_detection", {})}
        opto["lag_estimation"] = {**DEFAULT_LAG_ESTIMATION, **opto.get("lag_estimation", {})}
        opto["artifact_mask"] = {**DEFAULT_ARTIFACT_MASK, **opto.get("artifact_mask", {})}

    meta["_path"] = str(path)
    return meta


def load_merge_recipe(path) -> dict:
    """
    Load and validate a merge recipe YAML, resolving each ``sessions[].metadata``
    (relative to the recipe's own directory) into a full session dict via
    ``load_session_metadata``.

    Returns the recipe dict with ``sessions`` replaced by the loaded session
    dicts, ``phase_shift`` filled with defaults, and ``_path`` added.
    """
    path = Path(path).resolve()
    recipe = _read_yaml(path)

    missing = [k for k in _REQUIRED_RECIPE_KEYS if k not in recipe]
    if missing:
        raise ValueError(f"{path.name}: missing required field(s) {missing}")

    if not isinstance(recipe["probe_streams"], list) or not recipe["probe_streams"]:
        raise ValueError(f"{path.name}: probe_streams must be a non-empty list")
    if not isinstance(recipe["sessions"], list) or not recipe["sessions"]:
        raise ValueError(f"{path.name}: sessions must be a non-empty list")

    recipe["phase_shift"] = {**_PHASE_SHIFT_DEFAULTS, **recipe.get("phase_shift", {})}
    recipe.setdefault("kilosort_hint", {})

    sessions = []
    for entry in recipe["sessions"]:
        if "metadata" not in entry:
            raise ValueError(f"{path.name}: every sessions[] entry needs 'metadata'")
        sessions.append(load_session_metadata(path.parent / entry["metadata"]))
    recipe["sessions"] = sessions

    # every probe stream the recipe wants must be declared by every session
    for probe in recipe["probe_streams"]:
        for s in sessions:
            if probe not in s["probe_streams"]:
                raise ValueError(
                    f"{path.name}: probe_stream {probe!r} not in probe_streams of "
                    f"session {s['session_id']!r}"
                )

    recipe["_path"] = str(path)
    return recipe
