"""
Raw (unfiltered) single-channel mean waveforms read straight from the binary
Kilosort ran on, waveform-shape features, and putative cell-type labels.

Ported from Dan's MATLAB pipeline (Dan-neuropixels-optotagging-pipeline/matlab):
extract_raw_mean_waveform_one_channel.m, waveform_p2v_features_upsampled_centered.m,
classify_waveform_box.m, add_raw_waveform_features_and_type_to_units_v2.m.
Binaries are only ever opened read-only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from spikeinterface.core import BaseSorting
from tqdm.auto import tqdm

DEFAULT_WAVEFORM_PARAMS = {
    "pre_ms": 1.0,          # snippet before / after the spike sample
    "post_ms": 2.0,
    "max_spikes": 500,      # evenly spaced spikes averaged per unit
    "min_spikes": 10,       # fewer spikes in the interval -> feature failed
    "interp_factor": 20,    # upsampling before measuring widths
}

DEFAULT_SPN_PARAMS = {      # Dan's pMSN gate (strict inequalities)
    "min_peak_width_us": 90.0,
    "min_p2v_us": 560.0,
    "max_fr_hz": 15.0,
}


def open_raw_binary(dat_path, n_chan: int) -> np.ndarray:
    """Read-only int16 memmap of a sample-major binary, shaped (n_samples, n_chan)."""
    return np.memmap(Path(dat_path), dtype="int16", mode="r").reshape(-1, int(n_chan))


def evenly_spaced(values: np.ndarray, max_n: int) -> np.ndarray:
    """At most max_n entries of values, evenly spaced (MATLAB round(linspace(1, n, max_n)))."""
    if values.size <= max_n:
        return values
    return values[np.round(np.linspace(0, values.size - 1, max_n)).astype(int)]


def mean_raw_waveform(dat: np.ndarray, spike_samples: np.ndarray, channel: int,
                      n_pre: int, n_post: int, bit_volts: float):
    """
    Mean raw snippet (uV) on one channel, samples [s - n_pre, s + n_post] around
    each spike sample s; spikes too close to either end are skipped.

    Returns (mean_waveform or None, number of spikes averaged).
    """
    total, n_kept = np.zeros(n_pre + n_post + 1), 0
    for s in spike_samples:
        if s - n_pre < 0 or s + n_post >= dat.shape[0]:
            continue
        total += dat[s - n_pre:s + n_post + 1, channel]
        n_kept += 1
    return (total / n_kept * bit_volts, n_kept) if n_kept else (None, 0)


def waveform_features(wave: np.ndarray, fs: float, pre_ms: float, post_ms: float,
                      interp_factor: int = 20) -> dict:
    """
    Peak width and peak-to-valley of a mean waveform, port of
    waveform_p2v_features_upsampled_centered.m.

    The waveform (pre_ms before to post_ms after the spike sample) is baseline
    subtracted (median of the samples up to -1 ms), upsampled by interp_factor,
    and oriented so its main deflection within [-0.5, +1] ms is positive.
    ``peak_width_us`` is the full width at half amplitude of that deflection
    (the classic spike "half-width"); ``p2v_us`` is the time from it to the
    opposite-polarity extremum within the following 3 ms.

    Returns a dict: peak_width_us, p2v_us, peak_val, valley_val, t_peak_us,
    t_valley_us, feature_failed, fail_reason.
    """
    out = {"peak_width_us": np.nan, "p2v_us": np.nan, "peak_val": np.nan, "valley_val": np.nan,
           "t_peak_us": np.nan, "t_valley_us": np.nan, "feature_failed": True, "fail_reason": ""}

    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size == 0 or not np.isfinite(wave).any():
        out["fail_reason"] = "empty_or_bad_wave"
        return out
    wave = np.where(np.isfinite(wave), wave, np.median(wave[np.isfinite(wave)]))

    t_native_us = (np.arange(wave.size) - round(pre_ms / 1000.0 * fs)) / fs * 1e6
    base = (t_native_us >= -pre_ms * 1000.0) & (t_native_us <= -1000.0)
    wave = wave - np.median(wave[base] if base.any() else wave)

    w_up = resample_poly(wave, interp_factor, 1)
    t_us = t_native_us[0] + np.arange(w_up.size) / (fs * interp_factor) * 1e6

    center = np.flatnonzero((t_us >= -500.0) & (t_us <= 1000.0))
    if center.size == 0:
        out["fail_reason"] = "no_center_window"
        return out
    main = center[np.argmax(np.abs(w_up[center]))]
    sign = np.sign(w_up[main])
    if sign == 0:
        out["fail_reason"] = "zero_main_deflection"
        return out
    w = sign * w_up                                         # main deflection positive

    pk = center[np.argmax(w[center])]
    out["t_peak_us"], out["peak_val"] = float(t_us[pk]), float(w[pk])
    if w[pk] <= 0:
        out["fail_reason"] = "nonpositive_peak"
        return out

    half = w[pk] / 2.0
    left = np.flatnonzero((t_us < t_us[pk]) & (w < half))
    if left.size == 0:
        out["fail_reason"] = "no_left_half_crossing"
        return out
    right = np.flatnonzero((np.arange(w.size) > pk) & (w < half))
    if right.size == 0:
        out["fail_reason"] = "no_right_half_crossing"
        return out
    i1, i2 = left[-1], right[0]
    if i2 <= i1:
        out["fail_reason"] = "bad_half_crossing_order"
        return out
    out["peak_width_us"] = float(t_us[i2] - t_us[i1])

    post = np.flatnonzero((t_us > t_us[pk]) & (t_us <= t_us[pk] + 3000.0))
    if post.size == 0:
        out["fail_reason"] = "no_post_peak_window"
        return out
    v = post[np.argmin(w[post])]
    out["t_valley_us"], out["valley_val"] = float(t_us[v]), float(w[v])
    out["p2v_us"] = float(t_us[v] - t_us[pk])

    if not (np.isfinite(out["peak_width_us"]) and out["peak_width_us"] > 0):
        out["fail_reason"] = "bad_peak_width"
        return out
    if not (np.isfinite(out["p2v_us"]) and out["p2v_us"] > 0):
        out["fail_reason"] = "bad_p2v"
        return out
    out["feature_failed"] = False
    return out


def classify_waveform_box(peak_width_us: float, p2v_us: float) -> str:
    """Dan's waveform boxes (classify_waveform_box.m): "FSI", "MSN", "O", or "NA"."""
    if not (np.isfinite(peak_width_us) and np.isfinite(p2v_us)):
        return "NA"
    if 50 < peak_width_us < 200 and 100 < p2v_us < 455:
        return "FSI"
    if 150 < peak_width_us < 450 and 560 < p2v_us < 1500:
        return "MSN"
    if 200 < peak_width_us < 300 and 300 < p2v_us < 550:
        return "O"
    return "NA"


def compute_unit_waveform_features(sorting: BaseSorting, dat_path, n_chan: int, *,
                                   spike_interval=None, params: dict | None = None,
                                   bit_volts: float = 0.195, progress_bar: bool = True,
                                   return_waveforms: bool = False):
    """
    Per unit: raw mean waveform on its Phy peak channel (``ch`` property) and
    its waveform features, as in add_raw_waveform_features_and_type_to_units_v2.m.

    Parameters
    ----------
    sorting : spikeinterface BaseSorting
        With a ``ch`` property (Phy cluster_info.tsv; dat channel number).
    dat_path : Path
        The binary Kilosort ran on (see src.optotagging.read_kilosort_params).
        Opened read-only.
    n_chan : int
        Channels per sample in dat_path.
    spike_interval : (start_sample, end_sample) | None
        Only spikes in [start, end) are used (e.g. the behavior session, to
        keep laser-evoked spikes out); None = all spikes.
    params : dict | None
        Overrides for DEFAULT_WAVEFORM_PARAMS.
    bit_volts : float
        int16 -> uV for the returned waveforms (features are scale-invariant).
    progress_bar : bool
        Show a per-unit progress bar.
    return_waveforms : bool
        Also return {unit_id: (t_ms, mean_waveform_uV)}.

    Returns
    -------
    pandas.DataFrame indexed by unit_id (ascending): ch, n_wf_spikes, the
    waveform_features fields, waveform_type  [, waveforms dict]
    """
    params = {**DEFAULT_WAVEFORM_PARAMS, **(params or {})}
    if "ch" not in sorting.get_property_keys():
        raise ValueError("sorting has no 'ch' property (Phy cluster_info.tsv) for the peak channel")
    fs = sorting.get_sampling_frequency()
    dat = open_raw_binary(dat_path, n_chan)
    n_pre = int(round(params["pre_ms"] / 1000.0 * fs))
    n_post = int(round(params["post_ms"] / 1000.0 * fs))
    channels = dict(zip(sorting.unit_ids, sorting.get_property("ch")))

    rows, waveforms = [], {}
    for unit_id in tqdm(sorted(sorting.unit_ids), desc="raw waveforms", unit="unit",
                        disable=not progress_bar):
        spikes = sorting.get_unit_spike_train(unit_id).astype(np.int64)
        if spike_interval is not None:
            spikes = spikes[(spikes >= spike_interval[0]) & (spikes < spike_interval[1])]
        channel = int(channels[unit_id])
        row = {"unit_id": unit_id, "ch": channel, "n_wf_spikes": 0}

        wave, n_used = (None, 0)
        if spikes.size >= params["min_spikes"]:
            wave, n_used = mean_raw_waveform(dat, evenly_spaced(spikes, params["max_spikes"]),
                                             channel, n_pre, n_post, bit_volts)
        row["n_wf_spikes"] = n_used
        if wave is None:
            features = waveform_features(np.array([]), fs, params["pre_ms"], params["post_ms"])
            features["fail_reason"] = f"fewer than {params['min_spikes']} spikes in waveform interval"
        else:
            features = waveform_features(wave, fs, params["pre_ms"], params["post_ms"],
                                         params["interp_factor"])
            waveforms[unit_id] = ((np.arange(wave.size) - n_pre) / fs * 1000.0, wave)
        row.update(features)
        row["waveform_type"] = classify_waveform_box(row["peak_width_us"], row["p2v_us"])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("unit_id")
    return (df, waveforms) if return_waveforms else df


def classify_putative_spn(features: pd.DataFrame, *, params: dict | None = None) -> pd.DataFrame:
    """
    Putative SPN (pMSN) gate from add_raw_waveform_features_and_type_to_units_v2.m:
    features not failed & peak_width_us > min_peak_width_us & p2v_us > min_p2v_us
    & firing_rate_hz < max_fr_hz.

    ``features`` needs peak_width_us, p2v_us, feature_failed, firing_rate_hz.
    Returns a copy with ``is_spn`` and ``spn_fail_reason`` (first failing
    criterion: "waveform feature failed", "peak width", "peak-to-valley",
    "firing rate"; "" if it passes).
    """
    params = {**DEFAULT_SPN_PARAMS, **(params or {})}
    out = features.copy()
    ok_wf = ~out["feature_failed"].astype(bool)
    ok_width = out["peak_width_us"] > params["min_peak_width_us"]
    ok_p2v = out["p2v_us"] > params["min_p2v_us"]
    ok_fr = out["firing_rate_hz"] < params["max_fr_hz"]
    out["is_spn"] = ok_wf & ok_width & ok_p2v & ok_fr
    reason = np.select([~ok_wf, ~ok_width, ~ok_p2v, ~ok_fr],
                       ["waveform feature failed", "peak width", "peak-to-valley", "firing rate"], "")
    out["spn_fail_reason"] = reason
    return out
