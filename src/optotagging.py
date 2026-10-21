"""
Loading a Phy-curated Kilosort 4 sorting (via SpikeInterface) and slicing it
back into the individual sessions that were concatenated before sorting
(see src/preprocess.py write_concatenated_recording and
notebook/2.2_concatenate_sessions_si.ipynb's session_boundaries.json), plus
laser-response / optotagging classification (SALT test).
"""

import ast
import operator
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si
from spikeinterface.core import BaseSorting
from tqdm.auto import tqdm

from src.waveform import evenly_spaced, mean_raw_waveform, open_raw_binary

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
                                        gap_factor: float = 3.0, fs: float | None = None,
                                        width_tol_ms: float = 0.5,
                                        freq_tol_frac: float = 0.05) -> pd.DataFrame:
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
    fs : float | None
        Sampling rate of on/off_sample_concat. When given, each block's measured
        pulse width (median off - on) and rate (1 / median inter-pulse interval)
        are checked against the protocol; None skips these checks.
    width_tol_ms, freq_tol_frac : float
        Tolerances for those checks (a mismatch only warns).

    Returns
    -------
    pandas.DataFrame
        Copy of laser_events with added columns: block (0-based index within
        colour), power (note_power string), current, protocol_frequency_hz,
        pulse_width_ms (the protocol's nominal pulse width).

    Raises
    ------
    ValueError
        If a colour's detected block count doesn't match the protocol, or a
        block's pulse count differs from the protocol's `count`.
    """
    out = laser_events.copy()
    out["block"] = -1
    out["power"] = None
    out["current"] = np.nan
    out["protocol_frequency_hz"] = np.nan
    out["pulse_width_ms"] = np.nan

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
            if n_pulses != row["count"]:
                raise ValueError(
                    f"{ev_color} block {bi}: {n_pulses} pulses vs protocol count {row['count']}"
                )
            idx = g.index[start_idx:end_idx]

            if fs is not None:
                block = g.loc[idx]
                width_ms = float(np.median(block["off_sample_concat"] - block["on_sample_concat"])) / fs * 1000.0
                if abs(width_ms - row["pulse_width"]) > width_tol_ms:
                    warnings.warn(f"{ev_color} block {bi}: measured pulse width {width_ms:.3f} ms vs "
                                  f"protocol {row['pulse_width']} ms")
                if n_pulses >= 2:
                    rate_hz = fs / float(np.median(np.diff(block["on_sample_concat"].to_numpy())))
                    if abs(rate_hz - row["frequency"]) > freq_tol_frac * row["frequency"]:
                        warnings.warn(f"{ev_color} block {bi}: measured rate {rate_hz:.3f} Hz vs "
                                      f"protocol {row['frequency']} Hz")

            out.loc[idx, "block"] = bi
            out.loc[idx, "power"] = row["note_power"]
            out.loc[idx, "current"] = row["current"]
            out.loc[idx, "protocol_frequency_hz"] = row["frequency"]
            out.loc[idx, "pulse_width_ms"] = float(row["pulse_width"])

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
# which this port deliberately does not reproduce. The baseline / test windows
# follow compute_laser_window_metrics_v2.m (see compute_laser_response).


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
    first_edge, last_edge = min(baseline_edges[0], test_edges[0]), max(baseline_edges[-1], test_edges[-1])
    for i, t0 in enumerate(pulse_times_s):
        # only the spikes that can land in either raster (spike_times_s is sorted)
        lo, hi = np.searchsorted(spike_times_s, [t0 + first_edge, t0 + last_edge], side="left")
        rel = spike_times_s[lo:hi + 1] - t0
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


def _pairwise_js_sqrt_divergence(histograms: np.ndarray) -> np.ndarray:
    """
    sqrt(KL(p, m) + KL(q, m)), m = (p + q) / 2, for every pair of rows of
    ``histograms`` (salt_DE.m's sqrt(2 * JSdiv); KL with eps in the
    denominator as in its KLdist). Returns a (k, k) matrix. m > 0 wherever p > 0
    or q > 0, so KL is always finite.
    """
    p = histograms[:, None, :]
    q = histograms[None, :, :]
    m = 0.5 * (p + q)
    eps = np.finfo(float).eps
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log(p / (m + eps)), 0.0).sum(axis=2)
        kl_q = np.where(q > 0, q * np.log(q / (m + eps)), 0.0).sum(axis=2)
    # clip to 0: KL(p,m)+KL(q,m) is mathematically >= 0, but floating-point
    # cancellation can make it a tiny negative number when p ~= q, which would
    # otherwise make sqrt() emit nan
    return np.sqrt(np.maximum(0.0, kl_p + kl_q))


def salt_test(spike_times_s, pulse_times_s, *, baseline_start_s: float, baseline_end_s: float,
             test_start_s: float, win_s: float, bin_s: float = 0.001):
    """
    Stimulus-associated spike latency test (Kvitsiani et al. 2013), ported
    from salt_DE.m.

    Splits the baseline span into consecutive, non-overlapping ``win_s``-wide
    windows (kept nearest ``baseline_end_s``), builds a first-spike-latency
    histogram for each one plus for the single test window, measures pairwise
    distances between every pair of histograms (sqrt(2x Jensen-Shannon
    divergence)), and compares the mean test-vs-baseline distance against the
    null distribution of baseline-vs-baseline distances.

    Parameters
    ----------
    spike_times_s, pulse_times_s : array-like, seconds
    baseline_start_s, baseline_end_s : float
        Available baseline span, in seconds relative to each pulse onset.
    test_start_s : float
        Start of the test window relative to each pulse onset (e.g. 0.001 for
        a response window starting 1 ms after onset).
    win_s, bin_s : float
        SALT window / bin width, seconds (win_s = test-window length).

    Returns
    -------
    p_value, stat : float
        Empirical p-value and salt_DE.m's Idiff: mean test-vs-baseline distance
        minus mean baseline-vs-baseline distance.
    n_baseline_windows : int
        Number of complete baseline windows used.
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
    upper = np.triu_indices(kn, k=1)
    distances[upper] = _pairwise_js_sqrt_divergence(np.asarray(histograms))[upper]

    null_distances = distances[:kn - 1, :kn - 1]
    null_distances = null_distances[np.isfinite(null_distances)]
    test_distance = float(np.nanmean(distances[:kn - 1, kn - 1]))

    if null_distances.size == 0 or not np.isfinite(test_distance):
        return np.nan, np.nan, n_baseline_windows

    p_value = float(np.sum(null_distances >= test_distance) / null_distances.size)
    return p_value, test_distance - float(np.mean(null_distances)), n_baseline_windows


# --------------------------------------------------------------------------- #
# Per-unit laser response + optotagging classification
# --------------------------------------------------------------------------- #
#
# Mirrors Dan's MATLAB pipeline as run with config/analysis/opto_1hz_salt.json:
# plot_optotag_pdf_per_cell_v4.m -> compute_laser_window_metrics_v2.m
# (p_method "salt") -> build_salt_rasters_exact.m + salt_DE.m, plus the
# waveform-correlation check from extract_waveforms_from_raw_fast.m /
# plot_optotag_pdf_per_cell_v4.m. Two deliberate differences: pulse onsets are
# the lag-corrected ones from laser_events.csv, and the SALT baseline also
# keeps clear of the previous pulse's offset artifact mask.

DEFAULT_OPTOTAG_PARAMS = {
    "max_trials": 100,                     # first N pulses (in time) per (color, power) condition
    "response_start_ms": 1.0,              # response window = [response_start_ms, pulse width]
    "artifact_blank_ms": (0.0, 1.0),
    "salt_bin_ms": 1.0,
    "salt_baseline_guard_ms": 10.0,        # baseline ends this long before each pulse
    "post_off_ms": None,                   # laser-offset mask length after offset (preprocessing.json)
    "salt_alpha": 0.001,
    "min_reliability": 0.25,
    "min_responded": 10,
    "require_wave_corr": True,
    "min_wave_corr": 0.9,
    "wave_corr_window_ms": (-0.35, 0.9),   # relative to the reference waveform's trough
    "wave_evoked_window_ms": (1.0, 10.0),  # latency window for laser-evoked waveform spikes
    "wave_max_spikes": 100,
    "wave_n_pre": 20,                      # snippet samples before / after the spike sample
    "wave_n_post": 41,
}


def compute_laser_response(spike_times_s, pulse_times_s, *, pulse_width_ms: float,
                           post_off_ms: float, response_start_ms: float = 1.0,
                           artifact_blank_ms=(0.0, 1.0), salt_bin_ms: float = 1.0,
                           salt_baseline_guard_ms: float = 10.0) -> dict:
    """
    Per-unit laser response metrics for one set of same-condition pulses,
    ported from compute_laser_window_metrics_v2.m (first-spike latency /
    reliability / jitter) plus the salt_test above.

    Parameters
    ----------
    spike_times_s, pulse_times_s : array-like, seconds
    pulse_width_ms : float
        Nominal pulse width; the response window is
        [response_start_ms, pulse_width_ms] (inclusive) and the SALT test
        window is [response_start_ms, pulse_width_ms) in salt_bin_ms bins.
    post_off_ms : float
        Length of the laser-offset artifact mask after each pulse offset
        (preprocessing.json mask_laser_artifact.post_off_ms).
    response_start_ms : float
        Start of the response window, e.g. 1 ms to skip the onset mask.
    artifact_blank_ms : (start, end)
        Spikes whose latency falls in this interval are ignored (the first
        *un-blanked* spike in the response window is used instead); use (0, 0)
        to disable.
    salt_bin_ms : float
        SALT bin width.
    salt_baseline_guard_ms : float
        The baseline ends this long before each pulse, and starts at least this
        long after the previous one.

    The available SALT baseline is
    [-min_IPI + max(guard, response end, pulse_width + post_off_ms), -guard]:
    as in compute_laser_window_metrics_v2.m, except that the shortest
    inter-pulse interval (not the median) is used and the previous pulse's
    offset mask is avoided, so no baseline window overlaps any masked data.

    Returns
    -------
    dict with n_trials, n_responded, reliability, latency_ms, median_latency_ms,
    jitter_ms, salt_p, salt_stat, n_baseline_windows, baseline_start_ms,
    first_latency_ms (per-trial, ms, nan = no response).
    """
    spike_times_s = np.sort(np.asarray(spike_times_s, dtype=np.float64))
    pulse_times_s = np.sort(np.asarray(pulse_times_s, dtype=np.float64))
    n_trials = int(pulse_times_s.size)

    if pulse_width_ms <= response_start_ms:
        raise ValueError(f"pulse_width_ms ({pulse_width_ms}) must exceed response_start_ms "
                         f"({response_start_ms})")
    resp_start_s, resp_end_s = response_start_ms / 1000.0, pulse_width_ms / 1000.0
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
        # MATLAB std(x, 0): sample standard deviation
        "jitter_ms": float(np.std(valid, ddof=1)) if n_responded >= 2 else (0.0 if n_responded == 1 else np.nan),
        "salt_p": np.nan,
        "salt_stat": np.nan,
        "n_baseline_windows": 0,
        "baseline_start_ms": np.nan,
        "first_latency_ms": first_latency_ms,
    }

    if n_trials >= 5:
        win_ms = pulse_width_ms - response_start_ms
        min_ipi_ms = float(np.min(np.diff(pulse_times_s))) * 1000.0
        guard_after_previous_ms = max(salt_baseline_guard_ms, pulse_width_ms, pulse_width_ms + post_off_ms)
        baseline_start_ms = -min_ipi_ms + guard_after_previous_ms
        baseline_end_ms = -salt_baseline_guard_ms
        if baseline_end_ms - baseline_start_ms >= 2 * win_ms:
            p, stat, n_windows = salt_test(
                spike_times_s, pulse_times_s,
                baseline_start_s=baseline_start_ms / 1000.0, baseline_end_s=baseline_end_ms / 1000.0,
                test_start_s=resp_start_s, win_s=win_ms / 1000.0, bin_s=salt_bin_ms / 1000.0)
            result.update(salt_p=p, salt_stat=stat, n_baseline_windows=n_windows,
                          baseline_start_ms=baseline_end_ms - n_windows * win_ms)
        else:
            warnings.warn(f"not enough clean inter-pulse baseline: available "
                          f"{baseline_end_ms - baseline_start_ms:.3f} ms, need at least {2 * win_ms:.3f} ms")

    return result


def spike_tag_passes(response: dict, *, salt_alpha: float = 0.001,
                     min_reliability: float = 0.25, min_responded: int = 10) -> tuple[bool, bool]:
    """
    (salt_pass, response_pass) for one compute_laser_response result, per
    compute_laser_window_metrics_v2.m: SALT p-value below salt_alpha; enough
    reliable responses.
    """
    salt_pass = bool(np.isfinite(response["salt_p"]) and response["salt_p"] < salt_alpha)
    response_pass = bool(np.isfinite(response["reliability"])
                         and response["reliability"] >= min_reliability
                         and response["n_responded"] >= min_responded)
    return salt_pass, response_pass


def is_optotagged(response: dict, *, salt_alpha: float = 0.001,
                  min_reliability: float = 0.25, min_responded: int = 10) -> bool:
    """
    Spike-time part of the opto-tag decision (SALT and response criteria both
    pass); the waveform check is applied separately in
    compute_population_opto_response.
    """
    return all(spike_tag_passes(response, salt_alpha=salt_alpha, min_reliability=min_reliability,
                                min_responded=min_responded))


# --------------------------------------------------------------------------- #
# Waveform check (session vs laser-evoked mean waveform)
# --------------------------------------------------------------------------- #

def read_kilosort_params(kilosort_path) -> dict:
    """
    Read a Kilosort/Phy ``params.py`` (plain literal assignments) without
    executing it. ``dat_path`` is resolved to an absolute Path (first entry,
    relative paths taken relative to the Kilosort folder).
    """
    kilosort_path = Path(kilosort_path)
    params = {}
    for node in ast.parse((kilosort_path / "params.py").read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            params[node.targets[0].id] = ast.literal_eval(node.value)
    dat_path = params["dat_path"]
    dat_path = Path(dat_path[0] if isinstance(dat_path, (list, tuple)) else dat_path)
    params["dat_path"] = dat_path if dat_path.is_absolute() else (kilosort_path / dat_path).resolve()
    return params


def _waveform_pearson(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Pearson r of x and y over mask (nan if fewer than 5 samples or zero variance)."""
    x, y = x[mask], y[mask]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid] - x[valid].mean(), y[valid] - y[valid].mean()
    if x.size < 5:
        return np.nan
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return float(np.sum(x * y) / denom) if denom > 0 else np.nan


def compute_waveform_correlation(sorting: BaseSorting, laser_events: pd.DataFrame, dat_path,
                                 n_chan: int, *, params: dict | None = None,
                                 bit_volts: float = 0.195, return_waveforms: bool = False,
                                 progress_bar: bool = True):
    """
    Waveform check from Dan's pipeline: Pearson correlation between each
    unit's reference mean waveform and its laser-evoked mean waveform, per
    laser colour, on the unit's Phy peak channel of the raw (unfiltered) binary.

    - Reference: up to ``wave_max_spikes`` evenly spaced spikes from all of the
      unit's spikes.
    - Evoked (per colour): spikes whose latency after the most recent pulse of
      that colour (all pulses of the colour, every power) lies in
      ``wave_evoked_window_ms`` (inclusive), up to ``wave_max_spikes`` evenly
      spaced.
    - r is computed over ``wave_corr_window_ms`` around the reference
      waveform's trough; ``wave_pass`` = r >= ``min_wave_corr`` (nan fails).

    Parameters
    ----------
    sorting : spikeinterface BaseSorting
        Concatenated-axis sorting with a ``ch`` property (Phy cluster_info.tsv).
    laser_events : pandas.DataFrame
        Pulses in the same sample coordinates (on_sample_concat), one probe.
    dat_path : Path
        The binary Kilosort ran on (see read_kilosort_params). Opened read-only.
    n_chan : int
        Channels per sample in dat_path.
    params : dict | None
        Overrides for DEFAULT_OPTOTAG_PARAMS (wave_* keys, min_wave_corr).
    bit_volts : float
        int16 -> uV (r is scale-invariant; only affects returned waveforms).
    return_waveforms : bool
        Also return {unit_id: {"t_ms": ..., "reference": ..., <color>: ...}}.
    progress_bar : bool
        Show a per-unit progress bar.

    Returns
    -------
    pandas.DataFrame, one row per (unit_id, color), sorted by unit_id:
    wave_channel, n_wave_ref, n_wave_evoked, wave_corr, wave_pass  [, waveforms dict]
    """
    params = {**DEFAULT_OPTOTAG_PARAMS, **(params or {})}
    if "ch" not in sorting.get_property_keys():
        raise ValueError("sorting has no 'ch' property (Phy cluster_info.tsv) for the peak channel")

    fs = sorting.get_sampling_frequency()
    dat = open_raw_binary(dat_path, n_chan)
    n_pre, n_post = params["wave_n_pre"], params["wave_n_post"]
    max_n = params["wave_max_spikes"]
    evoked_lo_ms, evoked_hi_ms = params["wave_evoked_window_ms"]

    pulses_by_color = {c: np.sort(g["on_sample_concat"].to_numpy(dtype=np.int64))
                       for c, g in laser_events.groupby("color")}
    channels = dict(zip(sorting.unit_ids, sorting.get_property("ch")))

    rows, waveforms = [], {}
    for unit_id in tqdm(sorted(sorting.unit_ids), desc="waveform check", unit="unit",
                        disable=not progress_bar):
        spikes = sorting.get_unit_spike_train(unit_id).astype(np.int64)
        channel = int(channels[unit_id])
        reference, n_ref = mean_raw_waveform(dat, evenly_spaced(spikes, max_n), channel, n_pre, n_post, bit_volts)
        if reference is not None:
            t_ms = (np.arange(reference.size) - int(np.argmin(reference))) / fs * 1000.0
            corr_mask = (t_ms >= params["wave_corr_window_ms"][0]) & (t_ms <= params["wave_corr_window_ms"][1])
            waveforms[unit_id] = {"t_ms": t_ms, "reference": reference}

        for color, pulses in pulses_by_color.items():
            k = np.searchsorted(pulses, spikes, side="right") - 1      # most recent pulse of this colour
            has_prev = k >= 0
            latency_ms = np.full(spikes.size, np.nan)
            latency_ms[has_prev] = (spikes[has_prev] - pulses[k[has_prev]]) / fs * 1000.0
            evoked = spikes[has_prev & (latency_ms >= evoked_lo_ms) & (latency_ms <= evoked_hi_ms)]

            wave_corr, n_evoked = np.nan, 0
            if reference is not None:
                evoked_mean, n_evoked = mean_raw_waveform(dat, evenly_spaced(evoked, max_n), channel,
                                                      n_pre, n_post, bit_volts)
                if evoked_mean is not None:
                    wave_corr = _waveform_pearson(reference, evoked_mean, corr_mask)
                    waveforms[unit_id][color] = evoked_mean
            rows.append({"unit_id": unit_id, "color": color, "wave_channel": channel,
                         "n_wave_ref": n_ref, "n_wave_evoked": n_evoked, "wave_corr": wave_corr,
                         "wave_pass": bool(np.isfinite(wave_corr) and wave_corr >= params["min_wave_corr"])})

    df = pd.DataFrame(rows)
    return (df, waveforms) if return_waveforms else df


def compute_population_opto_response(sorting: BaseSorting, laser_events: pd.DataFrame, *,
                                     params: dict | None = None, wave_df: pd.DataFrame | None = None,
                                     by=("color", "power"), verbose: bool = True,
                                     progress_bar: bool = True) -> pd.DataFrame:
    """
    Run compute_laser_response for every (unit, pulse group) and classify.

    Pulses are grouped by ``by`` — by default (color, power), so **each laser
    power level is tested for significance separately** — and only the first
    ``max_trials`` pulses (in time) of each group are used.

    A (unit, group) is tagged when SALT passes, the response criteria pass, and
    (if ``require_wave_corr``) the unit's waveform check for that colour passes.

    Parameters
    ----------
    sorting : spikeinterface BaseSorting
        Spike trains in the same sample-index coordinate system as
        laser_events' on_sample_concat (e.g. the unsliced, concatenated-axis
        sorting from load_curated_sorting — not a per-session frame_slice).
    laser_events : pandas.DataFrame
        From load_laser_events after annotate_laser_events_with_protocol
        (needs ``power`` and ``pulse_width_ms``).
    params : dict | None
        Overrides for DEFAULT_OPTOTAG_PARAMS; ``post_off_ms`` must be given.
    wave_df : pandas.DataFrame | None
        From compute_waveform_correlation; required if ``require_wave_corr``.
    by : sequence of str
        laser_events columns to group pulses by (missing columns are dropped).
    verbose : bool
        Print each group's pulse count, rate, and width.
    progress_bar : bool
        Show a progress bar over all (group, unit) pairs.

    Returns
    -------
    pandas.DataFrame, one row per (unit_id, <by...>), sorted by unit_id (groups
    in laser-delivery order within a unit): every compute_laser_response
    field, pulse_width_ms, salt_pass, response_pass, the waveform columns,
    ``is_tagged`` (this group passed), and ``color_tagged`` — True if the unit
    is tagged for at least one group sharing this colour.
    """
    params = {**DEFAULT_OPTOTAG_PARAMS, **(params or {})}
    if params["post_off_ms"] is None:
        raise ValueError("params['post_off_ms'] is required (preprocessing.json "
                         "probes.<probe>.mask_laser_artifact.post_off_ms)")
    if "pulse_width_ms" not in laser_events.columns:
        raise ValueError("laser_events has no pulse_width_ms column — run annotate_laser_events_with_protocol")
    if params["require_wave_corr"] and wave_df is None:
        raise ValueError("require_wave_corr is True but no wave_df was given")
    fs = sorting.get_sampling_frequency()

    by = [c for c in by if c in laser_events.columns]
    if not by:
        raise ValueError("laser_events has none of the requested `by` columns")

    spike_trains = {u: sorting.get_unit_spike_train(u).astype(np.float64) / fs
                    for u in sorting.unit_ids}

    groups = []   # (group_key, pulse_width_ms, pulse_times_s), in laser-delivery order
    for group_key, group_events in laser_events.groupby(by, sort=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        group_events = group_events.sort_values("on_sample_concat")
        widths = group_events["pulse_width_ms"].unique()
        if len(widths) != 1 or not np.isfinite(widths[0]):
            raise ValueError(f"group {group_key}: expected one pulse width, got {widths}")
        pulse_width_ms = float(widths[0])
        pulse_times_s = group_events["on_sample_concat"].to_numpy(dtype=np.float64)[:params["max_trials"]] / fs
        groups.append((group_key, pulse_width_ms, pulse_times_s))
        if verbose:
            rate_hz = 1.0 / np.median(np.diff(pulse_times_s)) if pulse_times_s.size >= 2 else np.nan
            print(f"{dict(zip(by, group_key))}: {len(group_events)} pulses, using {pulse_times_s.size} | "
                  f"rate {rate_hz:.3f} Hz | pulse width {pulse_width_ms:g} ms")
    groups.sort(key=lambda g: g[2][0])

    unit_ids = sorted(sorting.unit_ids)
    rows = []
    progress = tqdm(total=len(groups) * len(unit_ids), desc="laser response", unit="unit",
                    disable=not progress_bar)
    for unit_id in unit_ids:
        for group_key, pulse_width_ms, pulse_times_s in groups:
            progress.update()
            response = compute_laser_response(
                spike_trains[unit_id], pulse_times_s, pulse_width_ms=pulse_width_ms,
                post_off_ms=params["post_off_ms"], response_start_ms=params["response_start_ms"],
                artifact_blank_ms=params["artifact_blank_ms"], salt_bin_ms=params["salt_bin_ms"],
                salt_baseline_guard_ms=params["salt_baseline_guard_ms"])
            salt_pass, response_pass = spike_tag_passes(
                response, salt_alpha=params["salt_alpha"], min_reliability=params["min_reliability"],
                min_responded=params["min_responded"])
            rows.append({"unit_id": unit_id, **dict(zip(by, group_key)), "pulse_width_ms": pulse_width_ms,
                         "salt_pass": salt_pass, "response_pass": response_pass,
                         **{k: v for k, v in response.items() if k != "first_latency_ms"}})

    progress.close()

    df = pd.DataFrame(rows)
    if wave_df is not None:
        df = df.merge(wave_df[["unit_id", "color", "wave_corr", "wave_pass", "n_wave_evoked"]],
                      on=["unit_id", "color"], how="left")
        df["wave_pass"] = df["wave_pass"].fillna(False).astype(bool)
    wave_ok = df["wave_pass"] if params["require_wave_corr"] else True
    df["is_tagged"] = df["salt_pass"] & df["response_pass"] & wave_ok
    if "color" in df.columns:
        df["color_tagged"] = df.groupby(["unit_id", "color"])["is_tagged"].transform("any")
    else:
        df["color_tagged"] = df["is_tagged"]
    return df
