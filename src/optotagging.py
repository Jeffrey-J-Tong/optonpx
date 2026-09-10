"""
Loading a Phy-curated Kilosort 4 sorting (via SpikeInterface) and slicing it
back into the individual sessions that were concatenated before sorting
(see src/preprocess.py write_concatenated_recording and
notebook/2.2_concatenate_sessions_si.ipynb's session_boundaries.json).
"""

import operator
from pathlib import Path

import numpy as np
import spikeinterface.full as si
from spikeinterface.core import BaseSorting

DEFAULT_UNIT_CRITERIA = {"quality": ["good"]}

_COMPARISON_OPS = {
    ">": operator.gt, ">=": operator.ge,
    "<": operator.lt, "<=": operator.le,
    "==": operator.eq, "!=": operator.ne,
}


def _criterion_mask(values: np.ndarray, criterion) -> np.ndarray:
    """
    Evaluate one unit-property criterion against that property's per-unit values.

    ``criterion`` is either a bare value/list (shorthand for ``{"isin": ...}``)
    or a single-key dict naming the comparison: ">", ">=", "<", "<=", "==", "!=",
    "between" (inclusive range [lo, hi]), "outside" (outside an inclusive
    range), "isin", or "notin".
    """
    if not isinstance(criterion, dict):
        criterion = {"isin": criterion}
    if len(criterion) != 1:
        raise ValueError(f"a unit criterion must have exactly one operator, got {criterion}")
    (op_name, op_value), = criterion.items()

    if op_name in _COMPARISON_OPS:
        return _COMPARISON_OPS[op_name](values, op_value)
    if op_name == "between":
        lo, hi = op_value
        return (values >= lo) & (values <= hi)
    if op_name == "outside":
        lo, hi = op_value
        return (values < lo) | (values > hi)
    if op_name == "isin":
        op_value = op_value if isinstance(op_value, (list, tuple, set, np.ndarray)) else [op_value]
        return np.isin(values, list(op_value))
    if op_name == "notin":
        op_value = op_value if isinstance(op_value, (list, tuple, set, np.ndarray)) else [op_value]
        return ~np.isin(values, list(op_value))
    raise ValueError(f"unknown unit criterion operator {op_name!r}")


def load_curated_sorting(output_dir, probe_name: str, kilosort_dir: str = "kilosort4",
                         unit_criteria: dict | None = DEFAULT_UNIT_CRITERIA) -> BaseSorting:
    """
    Load a Kilosort 4 sorting via SpikeInterface's Phy reader, filtered by
    arbitrary per-unit criteria.

    Parameters
    ----------
    output_dir : Path
        The processed-session directory containing one subfolder per probe
        stream (as written by notebook 2.2_concatenate_sessions_si.ipynb).
    probe_name : str
        Probe stream subfolder name, e.g. "probeA".
    kilosort_dir : str
        Name of the Kilosort 4 results subfolder under
        ``output_dir/probe_name`` (or a shank_N subfolder of one) — Kilosort
        runs can land in differently named folders (e.g. "kilosort4",
        "kilosort4_no_car").
    unit_criteria : dict | None
        Maps a unit-property column (any key from ``sorting.get_property_keys()``,
        e.g. "quality", "fr", "ContamPct") to a criterion that column must
        satisfy — see ``_criterion_mask`` for the accepted forms. A bare value
        or list is shorthand for "isin", e.g. ``{"quality": ["good", "mua"]}``.
        A unit is kept only if it satisfies every criterion. None keeps every
        unit. Defaults to ``{"quality": ["good"]}`` (Phy-curated good units).

    Returns
    -------
    sorting : spikeinterface BaseSorting
        Spike trains are sample indices (sorting.get_unit_spike_train) in
        whatever coordinate system Kilosort was run on — the
        concatenated-recording coordinate system, if Kilosort was run on a
        concatenated binary. Every cluster_*.tsv column Kilosort/Phy wrote is
        loaded as a unit property (e.g. 'quality' = Phy's curated group,
        'KSLabel', 'Amplitude', 'ContamPct', 'depth', 'fr', 'sh', ...).
    """
    kilosort_path = Path(output_dir) / probe_name / kilosort_dir
    sorting = si.read_phy(kilosort_path, load_all_cluster_properties=True)

    if "quality" not in sorting.get_property_keys():
        raise ValueError(
            f"{kilosort_path}: no Phy-curated 'group' column found (only Kilosort's own "
            "KSLabel/metrics tsvs are present) — this sorting does not look like it has "
            "been manually curated in Phy yet."
        )

    if unit_criteria:
        mask = np.ones(len(sorting.unit_ids), dtype=bool)
        for column, criterion in unit_criteria.items():
            if column not in sorting.get_property_keys():
                raise ValueError(
                    f"unit_criteria references unknown column {column!r}; "
                    f"available columns: {sorting.get_property_keys()}"
                )
            mask &= _criterion_mask(sorting.get_property(column), criterion)
        sorting = sorting.select_units(sorting.unit_ids[mask])

    return sorting


def split_sorting_by_session(sorting: BaseSorting, sessions: list[dict]) -> dict:
    """
    Slice a concatenated-recording sorting back into per-session sortings.

    Parameters
    ----------
    sorting : spikeinterface BaseSorting
        A sorting whose spike trains are indices into a concatenated
        recording, as returned by load_curated_sorting.
    sessions : list[dict]
        Per-session boundary records with 'session_id', 'sample_start',
        'sample_end' — the "sessions" list under
        session_boundaries.json["streams"][probe_name], as written by
        notebook 2.2_concatenate_sessions_si.ipynb.

    Returns
    -------
    dict {session_id: BaseSorting}
        Each sorting is re-referenced to that session's own start (frame 0 =
        that session's sample_start), via SpikeInterface's frame_slice — as
        if Kilosort had been run on that session alone.
    """
    return {
        s["session_id"]: sorting.frame_slice(
            start_frame=int(s["sample_start"]), end_frame=int(s["sample_end"])
        )
        for s in sessions
    }
