"""
Preprocessing for Neuropixels 2.0 data recorded with Open Ephys: concatenate
multiple sessions into one Kilosort-ready binary, optionally masking optogenetic
light artifacts and applying the NP2.0 inter-channel phase-shift correction.

Pure numpy/scipy — no SpikeInterface. The raw recordings are only ever opened
read-only; every change lands in the output binary.

Processing order per session: mask -> phase shift -> append. Masking must precede
the phase shift (the FFT fractional delay would otherwise ring the sharp artifact
edges out past the narrow mask window), and every session is streamed on its own
so a phase-shift FFT block never straddles a session boundary.

Artifact masking (pulse detection, ADC->probe lag estimation, narrow linear-ramp
mask) reproduces the Berke lab optotagging pipeline
(matlab/load_laser_pulses_from_oe_adc.m, estimate_laser_artifact_lag_from_L.m,
mask_probe_dat_artifacts.m).
"""

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# NP2.0 inter-channel phase shift
# --------------------------------------------------------------------------- #

def npx2_sample_shifts(n_channels: int = 384) -> np.ndarray:
    """
    Per-channel sub-sample acquisition delay for a Neuropixels 2.0 probe, in
    units of samples (0 .. 15/16).

    NP2.0 multiplexes: each of its ADCs digitises a fixed group of channels one
    after another within a single sample tick, so channel c is sampled a little
    later than the nominal sample time. The offset depends only on the readout
    channel index:

        adc_sample_order[c] = (c // 2) % 16          # NP2.0: 16 samples / ADC cycle
        shift[c]            = adc_sample_order[c] / 16

    Cross-checked against probeinterface's ``mux_np2000`` table (24 channels per
    ADC x 16, LF rate 0 -> 16 cycles per ADC -> divide by 16).

    The sign convention matches SpikeInterface's ``phase_shift``: multiplying the
    channel spectrum by ``exp(-2j*pi*f*shift)`` (f in cycles/sample) advances the
    channel back onto the nominal sample grid.
    """
    ch = np.arange(int(n_channels))
    return ((ch // 2) % 16) / 16.0


def phase_shift_chunk(chunk: np.ndarray, shifts: np.ndarray,
                      left_margin: int = 0, right_margin: int = 0,
                      taper: bool = True) -> np.ndarray:
    """
    Shift each channel of ``chunk`` back by ``shifts[c]`` samples (sub-sample,
    via an FFT phase rotation) and return the interior, margins removed.

    Parameters
    ----------
    chunk : (n_channels, n_total) float array
        n_total = left_margin + keep + right_margin. The margins are context for
        the FFT (they suppress circular-convolution edge error) and are dropped
        from the result.
    shifts : (n_channels,) array
        Per-channel shift in samples (see ``npx2_sample_shifts``).
    left_margin, right_margin : int
        Sample counts to trim from each side after shifting.
    taper : bool
        Cosine-ramp the margin samples toward the per-channel mean before the
        FFT, so the (circular) transform does not see a hard step at the block
        edge. Only the discarded margins are tapered; the kept interior is not.

    Returns
    -------
    (n_channels, n_total - left_margin - right_margin) float64 array
    """
    import scipy.fft

    x = np.array(chunk, dtype=np.float64, copy=True)
    n_ch, n = x.shape
    means = x.mean(axis=1, keepdims=True)
    x -= means

    if taper:
        if left_margin > 0:
            r = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, left_margin, endpoint=False)))
            x[:, :left_margin] *= r
        if right_margin > 0:
            r = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0.0, right_margin, endpoint=False)))
            x[:, n - right_margin:] *= r

    freqs = np.fft.rfftfreq(n)  # cycles per sample
    rotation = np.exp(-2j * np.pi * freqs[None, :] * np.asarray(shifts, float)[:, None])
    shifted = scipy.fft.irfft(scipy.fft.rfft(x, axis=1) * rotation, n=n, axis=1)
    shifted += means

    return shifted[:, left_margin:n - right_margin]


# --------------------------------------------------------------------------- #
# Optogenetic laser pulse detection (from the OneBox ADC)
# --------------------------------------------------------------------------- #

DEFAULT_PULSE_DETECTION = {
    "threshold_volts": 2.0,
    "polarity": "positive",
    "min_width_ms": 5.0,
    "max_width_ms": 30.0,
    "min_interpulse_ms": 50.0,
}


def detect_laser_pulses(adc_volts: np.ndarray, adc_t: np.ndarray,
                        threshold_volts: float = 2.0, polarity: str = "positive",
                        min_width_ms: float = 5.0, max_width_ms: float = 30.0,
                        min_interpulse_ms: float = 50.0):
    """
    Detect laser pulse on/off times from one OneBox ADC channel.

    Port of ``load_laser_pulses_from_oe_adc.m``: subtract the median, threshold
    (``x > threshold_volts`` for positive polarity, ``x < threshold_volts`` for
    negative -- pass a negative threshold in that case), pair rising/falling
    edges, then filter by pulse width and by a minimum interval measured against
    the last *kept* pulse.

    Parameters
    ----------
    adc_volts : (n_samples,) array   -- one ADC channel, already scaled to volts
    adc_t     : (n_samples,) array   -- ADC timestamps in seconds (OE clock)

    Returns
    -------
    on_t, off_t : (n_pulses,) arrays of seconds (OE clock)
    """
    x = np.asarray(adc_volts, dtype=np.float64)
    x = x - np.median(x)
    adc_t = np.asarray(adc_t, dtype=np.float64)

    if polarity == "positive":
        is_pulse = x > threshold_volts
    elif polarity == "negative":
        is_pulse = x < threshold_volts
    else:
        raise ValueError("polarity must be 'positive' or 'negative'")

    is_pulse = is_pulse.astype(np.int8)
    on = np.flatnonzero(np.diff(is_pulse, prepend=np.int8(0)) == 1)
    off = np.flatnonzero(np.diff(is_pulse, append=np.int8(0)) == -1)
    k = min(on.size, off.size)
    on, off = on[:k], off[:k]
    valid = off > on
    on, off = on[valid], off[valid]

    on_t, off_t = adc_t[on], adc_t[off]
    width = off_t - on_t
    keep = (width >= min_width_ms / 1000.0) & (width <= max_width_ms / 1000.0)
    on_t, off_t = on_t[keep], off_t[keep]

    if on_t.size == 0:
        return on_t, off_t

    keep_idx = [0]
    last_kept = on_t[0]
    for i in range(1, on_t.size):
        if on_t[i] - last_kept >= min_interpulse_ms / 1000.0:
            keep_idx.append(i)
            last_kept = on_t[i]
    keep_idx = np.asarray(keep_idx)
    return on_t[keep_idx], off_t[keep_idx]


# --------------------------------------------------------------------------- #
# ADC -> probe artifact lag
# --------------------------------------------------------------------------- #

DEFAULT_LAG_ESTIMATION = {
    "probe_channel": 80,
    "search_window_ms": (-5.0, 8.0),
    "artifact_sign": "negative",
    "max_pulses": 200,
}


def estimate_artifact_lag(dat_path, probe_ts: np.ndarray, n_chan: int,
                          pulse_on_t: np.ndarray, probe_channel: int = 80,
                          search_window_ms=(-5.0, 8.0), artifact_sign: str = "negative",
                          max_pulses: int = 200):
    """
    Estimate the delay between a laser pulse onset (as timestamped on the ADC
    stream) and the artifact appearing in the probe data.

    Port of ``estimate_laser_artifact_lag_from_L.m``: for up to ``max_pulses``
    onsets, take the time of the extreme value (min for ``artifact_sign
    ='negative'``) on ``probe_channel`` within ``search_window_ms`` of the onset,
    and return the median.

    The probe binary is opened read-only. Result is scale-invariant, so no
    bit_volts conversion is needed.

    Parameters
    ----------
    dat_path       : path to the probe continuous.dat (int16, sample-major)
    probe_ts       : (n_samples,) probe timestamps in seconds (OE clock)
    n_chan         : channels per sample in dat_path
    pulse_on_t     : laser onset times, seconds (OE clock)
    probe_channel  : 0-indexed channel to read the artifact on

    Returns
    -------
    median_lag_ms : float
    per_pulse_ms  : (n_used,) array (may contain nan for skipped edge pulses)
    """
    probe_ts = np.asarray(probe_ts, dtype=np.float64)
    fs = 1.0 / np.median(np.diff(probe_ts))
    src = np.memmap(Path(dat_path), dtype="int16", mode="r").reshape(-1, int(n_chan))
    n_t = src.shape[0]

    pulses = np.asarray(pulse_on_t, dtype=np.float64)
    if pulses.size > max_pulses:
        pulses = pulses[np.round(np.linspace(0, pulses.size - 1, max_pulses)).astype(int)]

    n_pre = int(round(abs(search_window_ms[0]) / 1000.0 * fs))
    n_post = int(round(search_window_ms[1] / 1000.0 * fs))
    t_ms = (np.arange(-n_pre, n_post + 1) / fs) * 1000.0

    lags = np.full(pulses.size, np.nan)
    for k, pt in enumerate(pulses):
        c = int(np.searchsorted(probe_ts, pt))
        if c > 0 and (c >= n_t or (pt - probe_ts[c - 1]) < (probe_ts[c] - pt)):
            c -= 1
        i1, i2 = c - n_pre, c + n_post
        if i1 < 0 or i2 >= n_t:
            continue
        seg = src[i1:i2 + 1, int(probe_channel)].astype(np.float64)
        seg -= np.median(seg)
        ii = int(np.argmin(seg)) if artifact_sign == "negative" else int(np.argmax(seg))
        lags[k] = t_ms[ii]

    return float(np.nanmedian(lags)), lags


# --------------------------------------------------------------------------- #
# Mask windows + application
# --------------------------------------------------------------------------- #

DEFAULT_ARTIFACT_MASK = {
    "pre_on_ms": 0.1,
    "post_on_ms": 1.0,
    "pre_off_ms": 0.1,
    "post_off_ms": 1.0,
    "method": "linear",
}


def _nearest_sample(probe_ts: np.ndarray, t: float, n: int) -> int:
    c = int(np.searchsorted(probe_ts, t))
    if c > 0 and (c >= n or (t - probe_ts[c - 1]) < (probe_ts[c] - t)):
        c -= 1
    return min(max(c, 0), n - 1)


def build_mask_windows(on_t: np.ndarray, off_t: np.ndarray, probe_ts: np.ndarray,
                       fs: float, pre_on_ms: float = 0.1, post_on_ms: float = 1.0,
                       pre_off_ms: float = 0.1, post_off_ms: float = 1.0):
    """
    Turn laser on/off times into a merged list of ``(i1, i2)`` inclusive sample
    windows (session-local indices) to mask.

    Port of the event table in ``mask_probe_dat_artifacts.m``: one window around
    each onset and one around each offset, clipped to ``[1, n-2]``, then
    overlapping/adjacent windows merged.
    """
    probe_ts = np.asarray(probe_ts, dtype=np.float64)
    n = probe_ts.size

    events = [(float(t), pre_on_ms, post_on_ms) for t in np.atleast_1d(on_t)]
    events += [(float(t), pre_off_ms, post_off_ms) for t in np.atleast_1d(off_t)]

    wins = []
    for t, pre, post in events:
        c = _nearest_sample(probe_ts, t, n)
        i1 = max(1, c - int(round(pre / 1000.0 * fs)))
        i2 = min(n - 2, c + int(round(post / 1000.0 * fs)))
        if i2 >= i1:
            wins.append((i1, i2))

    wins.sort()
    merged = []
    for a, b in wins:
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def apply_ramp_mask(chunk: np.ndarray, chunk_abs_start: int, windows,
                    src_memmap: np.ndarray, method: str = "linear") -> None:
    """
    Replace the masked samples in ``chunk`` (modified in place).

    ``chunk`` is (n_channels, L) float covering absolute session samples
    ``[chunk_abs_start, chunk_abs_start + L)``. For ``method='linear'`` each
    window is filled with a per-channel straight line between the sample just
    before it and the sample just after it, both read from ``src_memmap`` by
    absolute index -- so the result does not depend on how the session was
    chunked. ``method='zero'`` writes zeros.

    ``src_memmap`` is the read-only source, shape (n_samples, n_channels).
    """
    n_ch, L = chunk.shape
    chunk_end = chunk_abs_start + L
    n_total = src_memmap.shape[0]

    for i1, i2 in windows:
        if i2 < chunk_abs_start or i1 >= chunk_end:
            continue
        j1 = max(i1, chunk_abs_start) - chunk_abs_start
        j2 = min(i2, chunk_end - 1) - chunk_abs_start

        if method == "zero":
            chunk[:, j1:j2 + 1] = 0.0
            continue
        if method != "linear":
            raise ValueError("method must be 'linear' or 'zero'")

        left = src_memmap[max(0, i1 - 1), :].astype(np.float64)
        right = src_memmap[min(n_total - 1, i2 + 1), :].astype(np.float64)
        full_len = i2 - i1 + 1
        # global 0..full_len-1 position of each sample we are about to write
        pos = np.arange(j1 + chunk_abs_start - i1, j2 + chunk_abs_start - i1 + 1)
        frac = pos / max(full_len - 1, 1)
        chunk[:, j1:j2 + 1] = left[:, None] + (right - left)[:, None] * frac[None, :]


# --------------------------------------------------------------------------- #
# Streaming concatenation
# --------------------------------------------------------------------------- #

def write_concatenated_recording(session_probe_infos, output_path, *,
                                 apply_phase_shift: bool = True, margin_ms: float = 40.0,
                                 chunk_s: float = 10.0, mask_method: str = "linear",
                                 progress=print):
    """
    Stream one probe stream of every session into a single Kilosort-ready int16
    binary at ``output_path``.

    ``session_probe_infos`` is a list (in concatenation order) of dicts:
        session_id     : str
        dat_path       : path to that session's probe continuous.dat  (read-only)
        source_session : name of the Open Ephys session folder (e.g.
                         "2026-09-04_14-20-43"; recorded in the result as
                         provenance only — never used to locate data, so it
                         stays valid even if the raw data is later moved to a
                         different drive/machine)
        n_samples      : int
        n_chan         : int
        fs             : float
        windows        : list of (i1, i2) session-local mask windows ([] to skip)
        mask_method    : "linear" or "zero" (optional; defaults to mask_method arg)

    Per session: read raw int16 in chunks (with margins for the phase shift),
    apply the ramp mask on a float copy, phase-shift, clip/round back to int16,
    append. The source files are opened ``mode='r'`` only.

    Returns
    -------
    per_session : list of dicts with session_id, source_session, source_dat_bytes,
                  sample_start, sample_end, n_samples  (indices into the output)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_session = []
    cursor = 0
    with open(output_path, "wb") as out:
        for info in session_probe_infos:
            n = int(info["n_samples"])
            n_ch = int(info["n_chan"])
            fs = float(info["fs"])
            windows = info.get("windows") or []
            method = info.get("mask_method", mask_method)

            src = np.memmap(Path(info["dat_path"]), dtype="int16", mode="r").reshape(-1, n_ch)
            if src.shape[0] != n:
                raise ValueError(
                    f"{info['session_id']}: dat has {src.shape[0]} samples but "
                    f"expected {n} (from timestamps.npy)"
                )

            shifts = npx2_sample_shifts(n_ch)
            chunk = max(1, int(round(chunk_s * fs)))
            margin = int(round(margin_ms / 1000.0 * fs)) if apply_phase_shift else 0

            for c0 in range(0, n, chunk):
                c1 = min(c0 + chunk, n)
                m0 = max(0, c0 - margin)
                m1 = min(n, c1 + margin)

                buf = np.ascontiguousarray(src[m0:m1, :].T).astype(np.float64)
                if windows:
                    apply_ramp_mask(buf, m0, windows, src, method)
                if apply_phase_shift:
                    buf = phase_shift_chunk(buf, shifts, c0 - m0, m1 - c1, taper=True)
                else:
                    buf = buf[:, (c0 - m0):(c0 - m0) + (c1 - c0)]

                out.write(np.clip(np.rint(buf.T), -32768, 32767).astype(np.int16).tobytes())
                progress(f"  {info['session_id']}: {c1}/{n} samples "
                         f"({100.0 * c1 / n:.0f}%)")

            per_session.append({
                "session_id": info["session_id"],
                "source_session": str(info["source_session"]),
                "source_dat_bytes": Path(info["dat_path"]).stat().st_size,
                "sample_start": cursor,
                "sample_end": cursor + n,
                "n_samples": n,
            })
            cursor += n

    return per_session
