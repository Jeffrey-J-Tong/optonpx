"""
Loading a Phy-curated Kilosort 4 sorting (via SpikeInterface) and slicing it
back into the individual sessions that were concatenated before sorting
(see src/preprocess.py write_concatenated_recording and
notebook/2.2_concatenate_sessions_si.ipynb's session_boundaries.json), plus
laser-response / optotagging classification (SALT test).
"""

import operator
from pathlib import Path

import numpy as np
import pandas as pd
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


# --------------------------------------------------------------------------- #
# Laser events
# --------------------------------------------------------------------------- #

def load_laser_events(output_dir, probe_name: str | None = None,
                      session_id: str | None = None) -> pd.DataFrame:
    """
    Load laser_events.csv (written by notebook 2.2_concatenate_sessions[_si]).

    Each row is one masked laser pulse: probe, session, color,
    on_sample_concat/off_sample_concat (sample index into the *concatenated*
    recording — the same coordinate system as an unsliced sorting's
    get_unit_spike_train), on_t_oe (pulse onset, seconds, on that session's own
    OE clock), lag_ms (ADC->probe artifact lag used to build the mask).

    Parameters
    ----------
    output_dir : Path
        The processed-session directory (same as load_curated_sorting's
        output_dir) containing laser_events.csv.
    probe_name, session_id : str | None
        If given, keep only rows matching that probe / session.

    Returns
    -------
    pandas.DataFrame
    """
    events = pd.read_csv(Path(output_dir) / "laser_events.csv")
    if probe_name is not None:
        events = events[events["probe"] == probe_name]
    if session_id is not None:
        events = events[events["session"] == session_id]
    return events.reset_index(drop=True)


_PROTOCOL_COLOR_TO_EVENT = {"blue": "blue_laser", "red": "red_laser"}


def load_laser_protocol(csv_path) -> pd.DataFrame:
    """
    Read a ``laser_control_protocol_*.csv`` — one row per stimulation block, in
    delivery order. Columns: color ("blue"/"red"), pulse_width, frequency,
    count, current (laser-driver mA), note_power (e.g. "5 mW"), note_wavelength.
    """
    return pd.read_csv(Path(csv_path))


def annotate_laser_events_with_protocol(laser_events: pd.DataFrame, protocol: pd.DataFrame, *,
                                        gap_factor: float = 3.0) -> pd.DataFrame:
    """
    Match each detected pulse block (per colour, in time order) to a row of
    ``protocol`` (per colour, in order) and tag every pulse with which block it
    belongs to and that block's power.

    Parameters
    ----------
    laser_events : pandas.DataFrame
        From load_laser_events (one probe, the optotagging session).
    protocol : pandas.DataFrame
        From load_laser_protocol.
    gap_factor : float
        Passed to laser_pulse_blocks for block detection.

    Returns
    -------
    pandas.DataFrame
        Copy of laser_events with added columns: block (0-based index within
        colour), power (note_power string), current, protocol_frequency_hz.

    Raises
    ------
    ValueError
        If a colour's detected block count doesn't match the protocol, or a
        block's pulse count is more than 10% off the protocol's `count`.
    """
    out = laser_events.copy()
    out["block"] = -1
    out["power"] = None
    out["current"] = np.nan
    out["protocol_frequency_hz"] = np.nan

    for ev_color in sorted(laser_events["color"].unique()):
        proto_color = next((k for k, v in _PROTOCOL_COLOR_TO_EVENT.items() if v == ev_color), ev_color)
        proto = protocol[protocol["color"] == proto_color].reset_index(drop=True)
        g = laser_events[laser_events["color"] == ev_color].sort_values("on_sample_concat")
        blocks = laser_pulse_blocks(g["on_sample_concat"].to_numpy(dtype=float), gap_factor=gap_factor)

        if len(blocks) != len(proto):
            raise ValueError(
                f"{ev_color}: detected {len(blocks)} pulse block(s) but the protocol has "
                f"{len(proto)} row(s) for colour {proto_color!r}"
            )
        for bi, (start_idx, end_idx, _) in enumerate(blocks):
            row = proto.iloc[bi]
            n_pulses = end_idx - start_idx
            if abs(n_pulses - row["count"]) > 0.1 * row["count"]:
                raise ValueError(
                    f"{ev_color} block {bi}: {n_pulses} pulses vs protocol count {row['count']}"
                )
            idx = g.index[start_idx:end_idx]
            out.loc[idx, "block"] = bi
            out.loc[idx, "power"] = row["note_power"]
            out.loc[idx, "current"] = row["current"]
            out.loc[idx, "protocol_frequency_hz"] = row["frequency"]

    return out


def laser_pulse_blocks(pulse_times_s, gap_factor: float = 3.0) -> list[tuple]:
    """
    Split a pulse train into blocks wherever the inter-pulse interval jumps to
    more than ``gap_factor`` times the median interval (e.g. an optotagging
    session made of several 1 Hz trains separated by long pauses, or blocks run
    at different rates).

    Parameters
    ----------
    pulse_times_s : array-like, seconds (ascending)
    gap_factor : float
        An interval longer than gap_factor x median(all intervals) starts a
        new block.

    Returns
    -------
    list of (start_idx, end_idx, rate_hz)
        Half-open index ranges into pulse_times_s, plus each block's pulse rate
        (1 / median within-block interval; nan for a 1-pulse block).
    """
    pulse_times_s = np.sort(np.asarray(pulse_times_s, dtype=np.float64))
    n = pulse_times_s.size
    if n < 2:
        return [(0, n, np.nan)]

    ipi = np.diff(pulse_times_s)
    boundaries = np.flatnonzero(ipi > gap_factor * np.median(ipi)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])

    blocks = []
    for s, e in zip(starts, ends):
        rate = 1.0 / np.median(np.diff(pulse_times_s[s:e])) if e - s >= 2 else np.nan
        blocks.append((int(s), int(e), float(rate)))
    return blocks


# --------------------------------------------------------------------------- #
# SALT (stimulus-associated spike latency test)
# --------------------------------------------------------------------------- #
#
# Port of salt_DE.m from the Berke-lab optotagging pipeline — itself a cleaned
# up version of the original Kepecs-lab SALT.m (Kvitsiani et al. 2013), with
# latency/jitter/reliability exports added by Ali Mohebi. This is the
# "standard" SALT variant (Dan's pipeline calls it salt_method="salt"), as
# opposed to Dan's own resampled_median_js / pooled_shuffle alternatives,
# which this port deliberately does not reproduce.

DEFAULT_SALT_PARAMS = {"win_ms": 10.0, "bin_ms": 1.0, "baseline_span_ms": 150.0,
                       "baseline_guard_ms": 10.0}


def _build_salt_rasters(spike_times_s: np.ndarray, pulse_times_s: np.ndarray, *,
                        baseline_start_s: float, baseline_end_s: float,
                        test_start_s: float, bin_s: float, win_s: float):
    """
    Port of build_salt_rasters_exact.m: per-trial boolean spike-presence
    rasters, one row per pulse, binned at ``bin_s`` resolution — a
    ``baseline_raster`` spanning as many complete, non-overlapping
    ``win_s``-wide windows as fit in ``[baseline_start_s, baseline_end_s)``
    (kept nearest ``baseline_end_s``, discarding any older incomplete
    remainder), and a single ``test_raster`` window at
    ``[test_start_s, test_start_s + win_s)``.

    Returns (baseline_raster, test_raster, n_baseline_windows, n_win_bins).
    """
    n_win_bins = int(round(win_s / bin_s))
    if abs(win_s / bin_s - n_win_bins) > 1e-8:
        raise ValueError(f"win_s ({win_s}) must be an integer number of bin_s ({bin_s}) bins")

    available_span_s = baseline_end_s - baseline_start_s
    if available_span_s <= 0:
        raise ValueError("no clean baseline interval available (baseline_end_s <= baseline_start_s)")
    n_available_bins = int(np.floor(available_span_s / bin_s + 1e-9))
    n_baseline_windows = n_available_bins // n_win_bins
    if n_baseline_windows < 2:
        raise ValueError(
            f"only {n_baseline_windows} complete baseline window(s) available "
            f"({available_span_s * 1000:.1f} ms); need at least 2"
        )
    n_baseline_bins = n_baseline_windows * n_win_bins

    # keep the baseline nearest the pulse; trim the older incomplete remainder
    baseline_start_used_s = baseline_end_s - n_baseline_bins * bin_s
    baseline_edges = baseline_start_used_s + np.arange(n_baseline_bins + 1) * bin_s
    test_edges = test_start_s + np.arange(n_win_bins + 1) * bin_s

    n_trials = pulse_times_s.size
    baseline_raster = np.zeros((n_trials, n_baseline_bins), dtype=bool)
    test_raster = np.zeros((n_trials, n_win_bins), dtype=bool)
    for i, t0 in enumerate(pulse_times_s):
        rel = spike_times_s - t0
        baseline_raster[i] = np.histogram(rel, baseline_edges)[0] > 0
        test_raster[i] = np.histogram(rel, test_edges)[0] > 0

    return baseline_raster, test_raster, n_baseline_windows, n_win_bins


def _first_spike_bin_per_window(raster: np.ndarray, n_bins_per_window: int) -> np.ndarray:
    """
    raster: (n_trials, n_windows * n_bins_per_window) boolean.
    Returns (n_trials, n_windows) int array: 1-indexed bin of the first spike
    within each window, or 0 if that window/trial has no spike.
    """
    n_trials, n_total_bins = raster.shape
    n_windows = n_total_bins // n_bins_per_window
    out = np.zeros((n_trials, n_windows), dtype=int)
    for w in range(n_windows):
        window = raster[:, w * n_bins_per_window:(w + 1) * n_bins_per_window]
        has_spike = window.any(axis=1)
        out[has_spike, w] = np.argmax(window[has_spike], axis=1) + 1
    return out


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    support = p > 0
    if np.any(q[support] <= 0):
        return np.inf
    return float(np.sum(p[support] * np.log(p[support] / (q[support] + np.finfo(float).eps))))


def _js_sqrt_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    # clip to 0: KL(p,m)+KL(q,m) is mathematically >= 0, but floating-point
    # cancellation can make it a tiny negative number when p ~= q, which would
    # otherwise make sqrt() emit nan
    jsd = max(0.0, _kl_divergence(p, m) + _kl_divergence(q, m))
    return float(np.sqrt(jsd))


def salt_test(spike_times_s, pulse_times_s, *, baseline_start_s: float, baseline_end_s: float,
             test_start_s: float, win_s: float = 0.01, bin_s: float = 0.001):
    """
    Stimulus-associated spike latency test (Kvitsiani et al. 2013), ported
    from salt_DE.m.

    Splits the baseline span into consecutive, non-overlapping ``win_s``-wide
    windows, builds a first-spike-latency histogram for each one plus for the
    single test window, measures pairwise distances between every pair of
    histograms (sqrt(2x Jensen-Shannon divergence)), and compares the mean
    test-vs-baseline distance against the null distribution of
    baseline-vs-baseline distances.

    Parameters
    ----------
    spike_times_s, pulse_times_s : array-like, seconds
    baseline_start_s, baseline_end_s : float
        Baseline span, in seconds relative to each pulse onset (e.g. -0.16 to
        -0.01 for a 150 ms baseline ending 10 ms before the pulse).
    test_start_s : float
        Start of the test window relative to each pulse onset (e.g. 0.001 for
        a response window starting 1 ms after onset).
    win_s, bin_s : float
        SALT window / bin width, seconds (default 10 ms / 1 ms).

    Returns
    -------
    p_value, stat : float
        Empirical p-value and the test statistic (mean test-vs-baseline
        distance minus mean baseline-vs-baseline distance would be Idiff in
        salt_DE.m; here ``stat`` is the raw mean test-vs-baseline distance).
    """
    spike_times_s = np.sort(np.asarray(spike_times_s, dtype=np.float64))
    pulse_times_s = np.asarray(pulse_times_s, dtype=np.float64)

    baseline_raster, test_raster, n_baseline_windows, n_win_bins = _build_salt_rasters(
        spike_times_s, pulse_times_s, baseline_start_s=baseline_start_s,
        baseline_end_s=baseline_end_s, test_start_s=test_start_s, bin_s=bin_s, win_s=win_s)

    baseline_lsi = _first_spike_bin_per_window(baseline_raster, n_win_bins)  # (n_trials, n_baseline_windows)
    test_lsi = _first_spike_bin_per_window(test_raster, n_win_bins)          # (n_trials, 1)

    n_trials = pulse_times_s.size
    histograms = []
    for w in range(n_baseline_windows):
        counts = np.bincount(baseline_lsi[:, w], minlength=n_win_bins + 1)[:n_win_bins + 1]
        histograms.append(counts / n_trials)
    test_counts = np.bincount(test_lsi[:, 0], minlength=n_win_bins + 1)[:n_win_bins + 1]
    histograms.append(test_counts / n_trials)

    kn = len(histograms)  # n_baseline_windows + 1 (the last one is the test window)
    distances = np.full((kn, kn), np.nan)
    for k1 in range(kn):
        for k2 in range(k1 + 1, kn):
            distances[k1, k2] = _js_sqrt_divergence(histograms[k1], histograms[k2])

    null_distances = distances[:kn - 1, :kn - 1]
    null_distances = null_distances[np.isfinite(null_distances)]
    test_distance = float(np.nanmean(distances[:kn - 1, kn - 1]))

    if null_distances.size == 0 or not np.isfinite(test_distance):
        return np.nan, np.nan

    p_value = float(np.sum(null_distances >= test_distance) / null_distances.size)
    return p_value, test_distance


# --------------------------------------------------------------------------- #
# Per-unit laser response + optotagging classification
# --------------------------------------------------------------------------- #

DEFAULT_OPTOTAG_PARAMS = {
    "response_window_ms": (1.0, 10.0),
    "artifact_blank_ms": (0.0, 1.0),
    "salt_params": DEFAULT_SALT_PARAMS,
    "salt_alpha": 0.001,
    "min_reliability": 0.25,
    "min_responded": 10,
}


def compute_laser_response(spike_times_s, pulse_times_s, *,
                           response_window_ms=(1.0, 10.0), artifact_blank_ms=(0.0, 1.0),
                           salt_params: dict | None = None) -> dict:
    """
    Per-unit laser response metrics for one set of same-color pulses, ported
    from compute_laser_window_metrics_v2.m (first-spike latency / reliability
    / jitter) plus the salt_test above.

    Parameters
    ----------
    spike_times_s, pulse_times_s : array-like, seconds
    response_window_ms : (start, end)
        Window after pulse onset to look for the first response spike in,
        e.g. (1, 10) ms — start at 1 ms to skip the masked artifact.
    artifact_blank_ms : (start, end)
        Spikes whose latency falls in this interval are ignored (the first
        *un-blanked* spike in the response window is used instead); use (0, 0)
        to disable.
    salt_params : dict | None
        Overrides for DEFAULT_SALT_PARAMS (win_ms, bin_ms, baseline_span_ms,
        baseline_guard_ms).

    Returns
    -------
    dict with n_trials, n_responded, reliability, latency_ms, median_latency_ms,
    jitter_ms, salt_p, salt_stat, first_latency_ms (per-trial, ms, nan = no response).
    """
    salt_params = {**DEFAULT_SALT_PARAMS, **(salt_params or {})}
    spike_times_s = np.sort(np.asarray(spike_times_s, dtype=np.float64))
    pulse_times_s = np.asarray(pulse_times_s, dtype=np.float64)
    n_trials = int(pulse_times_s.size)

    resp_start_s, resp_end_s = (np.asarray(response_window_ms, dtype=np.float64) / 1000.0)
    blank_start_s, blank_end_s = (np.asarray(artifact_blank_ms, dtype=np.float64) / 1000.0)

    first_latency_ms = np.full(n_trials, np.nan)
    for i, t0 in enumerate(pulse_times_s):
        idx = np.flatnonzero((spike_times_s >= t0 + resp_start_s) & (spike_times_s <= t0 + resp_end_s))
        if idx.size == 0:
            continue
        lat_s = spike_times_s[idx] - t0
        if blank_end_s > blank_start_s:
            lat_s = lat_s[(lat_s < blank_start_s) | (lat_s > blank_end_s)]
        if lat_s.size == 0:
            continue
        first_latency_ms[i] = lat_s[0] * 1000.0

    responded = ~np.isnan(first_latency_ms)
    n_responded = int(responded.sum())
    valid = first_latency_ms[responded]

    result = {
        "n_trials": n_trials,
        "n_responded": n_responded,
        "reliability": n_responded / n_trials if n_trials else np.nan,
        "latency_ms": float(np.mean(valid)) if n_responded else np.nan,
        "median_latency_ms": float(np.median(valid)) if n_responded else np.nan,
        "jitter_ms": float(np.std(valid)) if n_responded >= 2 else (0.0 if n_responded == 1 else np.nan),
        "salt_p": np.nan,
        "salt_stat": np.nan,
        "first_latency_ms": first_latency_ms,
    }

    if n_trials >= 5:
        baseline_end_s = -salt_params["baseline_guard_ms"] / 1000.0
        baseline_start_s = baseline_end_s - salt_params["baseline_span_ms"] / 1000.0
        try:
            result["salt_p"], result["salt_stat"] = salt_test(
                spike_times_s, pulse_times_s,
                baseline_start_s=baseline_start_s, baseline_end_s=baseline_end_s,
                test_start_s=resp_start_s,
                win_s=salt_params["win_ms"] / 1000.0, bin_s=salt_params["bin_ms"] / 1000.0)
        except ValueError:
            pass  # not enough clean baseline — leave salt_p/salt_stat as nan

    return result


def is_optotagged(response: dict, *, salt_alpha: float = 0.001,
                  min_reliability: float = 0.25, min_responded: int = 10) -> bool:
    """
    Classify one compute_laser_response result as opto-tagged: SALT p-value
    below salt_alpha AND enough reliable responses, per
    compute_laser_window_metrics_v2.m's Metr.is_tag_candidate.
    """
    salt_pass = np.isfinite(response["salt_p"]) and response["salt_p"] < salt_alpha
    response_pass = (np.isfinite(response["reliability"])
                     and response["reliability"] >= min_reliability
                     and response["n_responded"] >= min_responded)
    return bool(salt_pass and response_pass)


def compute_population_opto_response(sorting: BaseSorting, laser_events: pd.DataFrame, *,
                                     params: dict | None = None,
                                     by=("color", "power")) -> pd.DataFrame:
    """
    Run compute_laser_response + is_optotagged for every (unit, pulse group).

    Pulses are grouped by ``by`` — by default (color, power), so **each laser
    power level is tested for significance separately** (different neurons can
    respond at different powers, and pooling all powers dilutes both the SALT
    test and the reliability estimate).

    Parameters
    ----------
    sorting : spikeinterface BaseSorting
        Spike trains in the same sample-index coordinate system as
        laser_events' on_sample_concat (e.g. the unsliced, concatenated-axis
        sorting from load_curated_sorting — not a per-session frame_slice).
    laser_events : pandas.DataFrame
        As returned by load_laser_events, ideally after
        annotate_laser_events_with_protocol (which adds the ``power`` column).
    params : dict | None
        Overrides for DEFAULT_OPTOTAG_PARAMS (response_window_ms,
        artifact_blank_ms, salt_params, salt_alpha, min_reliability,
        min_responded).
    by : sequence of str
        laser_events columns to group pulses by. Entries missing from
        laser_events are dropped (so this still works on un-annotated events,
        falling back to ("color",)).

    Returns
    -------
    pandas.DataFrame, one row per (unit_id, <by...>): every compute_laser_response
    field, plus unit_id, the ``by`` columns, ``is_tagged`` (this group passed),
    and ``color_tagged`` — True if the unit is tagged for at least one group
    that shares this colour (i.e. "tagged for a colour if tagged at >= 1 of its
    power levels").
    """
    params = {**DEFAULT_OPTOTAG_PARAMS, **(params or {})}
    fs = sorting.get_sampling_frequency()

    by = [c for c in by if c in laser_events.columns]
    if not by:
        raise ValueError("laser_events has none of the requested `by` columns")

    spike_trains = {u: sorting.get_unit_spike_train(u).astype(np.float64) / fs
                    for u in sorting.unit_ids}

    rows = []
    for group_key, group_events in laser_events.groupby(by, sort=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        pulse_times_s = group_events["on_sample_concat"].to_numpy(dtype=np.float64) / fs
        for unit_id in sorting.unit_ids:
            response = compute_laser_response(
                spike_trains[unit_id], pulse_times_s,
                response_window_ms=params["response_window_ms"],
                artifact_blank_ms=params["artifact_blank_ms"],
                salt_params=params["salt_params"])
            tagged = is_optotagged(response, salt_alpha=params["salt_alpha"],
                                   min_reliability=params["min_reliability"],
                                   min_responded=params["min_responded"])
            rows.append({"unit_id": unit_id, **dict(zip(by, group_key)), "is_tagged": tagged,
                        **{k: v for k, v in response.items() if k != "first_latency_ms"}})

    df = pd.DataFrame(rows)
    if "color" in df.columns:
        df["color_tagged"] = df.groupby(["unit_id", "color"])["is_tagged"].transform("any")
    else:
        df["color_tagged"] = df["is_tagged"]
    return df
