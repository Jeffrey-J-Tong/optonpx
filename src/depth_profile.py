"""
Depth profile of a raw Neuropixels survey recording (notebook 1.2).

One survey recording = one shank recorded with a sparse full-length IMRO (384 channels spread
over 0-8.6 mm). Per channel this computes, from a time window of the raw data:

- AP-band noise (MAD of the 300-6000 Hz signal, no CAR),
- threshold crossings of the CAR-referenced AP signal (rate, amplitude, raster), with time bins
  where many channels cross together (global artifacts) removed,
- LFP power spectrum and band power (low-pass + decimation, Welch),
- LFP correlation between depths, with and without CAR.

The raw continuous.dat is opened read-only (np.memmap mode "r") and processed in chunks.
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from src.openephys import oe_parse_folders, oe_parse_params

DEFAULT_BANDS = {
    "delta 1-4 Hz": (1, 4),
    "theta 6-10 Hz": (6, 10),
    "beta 15-30 Hz": (15, 30),
    "gamma 30-80 Hz": (30, 80),
    "high gamma 80-200 Hz": (80, 200),
}

DEFAULT_PARAMS = dict(
    t_start_s=0.0,             # start of the analysed window, seconds from the first sample
    duration_s=60.0,           # length of the analysed window (clipped to the recording)
    chunk_s=5.0,               # processing chunk
    pad_s=1.0,                 # filter padding on each side of a chunk
    ap_band_hz=(300, 6000),
    threshold_mad=5.0,         # crossing threshold, in MADs of the CAR AP signal (negative peaks)
    refractory_ms=1.0,         # minimum distance between crossings on one channel
    artifact_bin_ms=10.0,      # bins where > artifact_fraction of the good channels cross
    artifact_fraction=0.3,     #   are treated as global artifacts and removed
    lfp_lowpass_hz=300.0,
    lfp_decimate=12,           # 30 kHz -> 2.5 kHz
    welch_segment_s=2.0,
    corr_band_hz=(1, 100),     # LFP band used for the depth-depth correlation
    bands=DEFAULT_BANDS,
)


def _mad(x):
    return np.median(np.abs(x - np.median(x, axis=0)), axis=0) / 0.6745


def compute_depth_profile(recording_dir, probe_name=None, **params):
    """
    Compute the depth profile of one probe stream of a survey recording.

    Parameters
    ----------
    recording_dir : Path — Open Ephys recording folder (contains 'Record Node ...')
    probe_name    : str | None — probe to use, matched case-insensitively to the end of the stream
                    name (e.g. "ProbeB" for "OneBox-100.ProbeB"); None = the only probe stream
    **params      : overrides of DEFAULT_PARAMS

    Returns
    -------
    dict of numpy arrays / scalars (see save_depth_profile); per-channel arrays are indexed by
    channel number, per-row arrays (`row_*`, `psd_rows`, `corr*`) by unique electrode row,
    sorted by ypos.
    """
    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    prm = {**DEFAULT_PARAMS, **params}

    recording_dir = Path(recording_dir)
    _, paths = oe_parse_folders(recording_dir)
    probes, _ = oe_parse_params(paths["xml"], paths["oebin"])
    def pick(items, name_of):
        """The single item matching probe_name (or the only item when probe_name is None)."""
        if probe_name is None:
            matches = items
        else:
            matches = [it for it in items if str(name_of(it)).split(".")[-1].lower() == probe_name.lower()]
        if len(matches) != 1:
            raise ValueError(f"{recording_dir}: probe_name={probe_name!r} must select exactly one probe "
                             f"stream; available: {[str(name_of(it)) for it in items]}")
        return matches[0]

    probe = pick(probes, lambda p: p["stream_name"])
    stream_dir = pick(paths["ephys_streams"], lambda p: p.name)
    fs = float(probe["sample_rate"])
    n_ch = int(probe["channel_count"])
    bit_volts = float(probe["channel_bit_volts"])
    channels = sorted(probe["channel_ypos"], key=int)
    ypos = np.array([float(probe["channel_ypos"][c]) for c in channels])
    xpos = np.array([float(probe["channel_xpos"][c]) for c in channels])
    shank = np.array([int(probe["channel_shank"][c]) for c in channels])
    electrode = np.array([int(probe["channel_electrode"][c]) for c in channels])

    dat_path = stream_dir / "continuous.dat"
    raw = np.memmap(dat_path, dtype=np.int16, mode="r").reshape(-1, n_ch)   # read-only
    n_total = raw.shape[0]
    s0 = int(round(prm["t_start_s"] * fs))
    if s0 >= n_total:
        raise ValueError(f"t_start_s={prm['t_start_s']} is beyond the recording ({n_total / fs:.1f} s)")
    s1 = min(n_total, s0 + int(round(prm["duration_s"] * fs)))
    dec = int(prm["lfp_decimate"])
    chunk = int(prm["chunk_s"] * fs) // dec * dec      # multiple of dec: LFP samples line up across chunks
    pad = int(prm["pad_s"] * fs)

    sos_ap = signal.butter(3, prm["ap_band_hz"], btype="band", fs=fs, output="sos")
    sos_lfp = signal.butter(4, prm["lfp_lowpass_hz"], btype="low", fs=fs, output="sos")

    first_int = np.asarray(raw[s0:min(s1, s0 + int(fs))])
    flat = np.ptp(first_int, axis=0) == 0              # constant channels (e.g. stuck at the rail)
    first = first_int.astype(np.float32) * bit_volts
    good = ~flat
    offset = np.median(first, axis=0)                  # fixed DC offset, so LFP chunks join without steps

    ap_mad_chunks, car_mad_chunks, lfp_chunks = [], [], []
    peak_t, peak_ch, peak_amp = [], [], []
    distance = max(1, int(prm["refractory_ms"] * 1e-3 * fs))
    for a in range(s0, s1, chunk):
        b = min(a + chunk, s1)
        lo, hi = max(0, a - pad), min(n_total, b + pad)
        x = np.asarray(raw[lo:hi], dtype=np.float32) * bit_volts - offset
        ap = signal.sosfiltfilt(sos_ap, x, axis=0)[a - lo:b - lo].astype(np.float32)
        lfp = signal.sosfiltfilt(sos_lfp, x, axis=0)[a - lo:b - lo:dec].astype(np.float32)
        del x
        ap_mad_chunks.append(_mad(ap))
        car = ap - np.median(ap[:, good], axis=1, keepdims=True)
        del ap
        car_mad = _mad(car)
        car_mad_chunks.append(car_mad)
        for c in np.flatnonzero(good):
            pk, props = signal.find_peaks(-car[:, c], height=prm["threshold_mad"] * car_mad[c],
                                          distance=distance)
            peak_t.append((pk + a - s0) / fs)
            peak_ch.append(np.full(len(pk), c))
            peak_amp.append(props["peak_heights"].astype(np.float32))
        del car
        lfp_chunks.append(lfp)
    peak_t, peak_ch, peak_amp = map(np.concatenate, (peak_t, peak_ch, peak_amp))
    duration = (s1 - s0) / fs

    # global artifacts: bins in which a large fraction of the good channels cross together
    bin_s = prm["artifact_bin_ms"] * 1e-3
    n_bins = int(np.ceil(duration / bin_s))
    peak_bin = np.minimum((peak_t / bin_s).astype(int), n_bins - 1)
    channels_per_bin = np.zeros(n_bins)
    pairs = np.unique(np.column_stack([peak_bin, peak_ch]), axis=0)
    np.add.at(channels_per_bin, pairs[:, 0], 1)
    artifact = channels_per_bin > prm["artifact_fraction"] * good.sum()
    artifact = artifact | np.roll(artifact, 1) | np.roll(artifact, -1)   # +/- one bin around each
    keep = ~artifact[peak_bin]
    clean_duration = duration - artifact.sum() * bin_s
    peak_t, peak_ch, peak_amp = peak_t[keep], peak_ch[keep], peak_amp[keep]

    spike_rate = np.bincount(peak_ch, minlength=n_ch) / clean_duration
    spike_amp = np.full(n_ch, np.nan)
    for c in np.flatnonzero(good):
        a_c = peak_amp[peak_ch == c]
        if len(a_c):
            spike_amp[c] = np.median(a_c)

    # LFP: spectrum per channel, then per electrode row (mean of the good channels on that row)
    lfp = np.concatenate(lfp_chunks)
    fs_lfp = fs / dec
    freqs, psd = signal.welch(lfp, fs=fs_lfp, nperseg=int(prm["welch_segment_s"] * fs_lfp), axis=0)
    psd = psd.T                                         # (n_ch, n_freq)
    band_names = list(prm["bands"])
    with np.errstate(divide="ignore"):              # flat channels have zero power
        band_power_db = np.array([10 * np.log10(psd[:, (freqs >= lo_f) & (freqs < hi_f)].mean(axis=1))
                                  for lo_f, hi_f in prm["bands"].values()])
    band_power_db[:, flat] = np.nan

    row_ypos = np.unique(ypos[good])
    row_members = [np.flatnonzero(good & (ypos == y)) for y in row_ypos]
    psd_rows = np.array([psd[m].mean(axis=0) for m in row_members])
    sos_corr = signal.butter(3, prm["corr_band_hz"], btype="band", fs=fs_lfp, output="sos")
    lfp_band = signal.sosfiltfilt(sos_corr, lfp, axis=0)
    lfp_band_car = lfp_band - np.median(lfp_band[:, good], axis=1, keepdims=True)
    corr = np.corrcoef(np.array([lfp_band[:, m].mean(axis=1) for m in row_members]))
    corr_car = np.corrcoef(np.array([lfp_band_car[:, m].mean(axis=1) for m in row_members]))

    return dict(
        recording=recording_dir.name, probe=stream_dir.name.split(".")[-1], params=prm, sample_rate=fs,
        duration_s=duration, clean_duration_s=clean_duration,
        channel=np.arange(n_ch), ypos=ypos, xpos=xpos, shank=shank, electrode=electrode, flat=flat,
        ap_mad_uv=np.where(flat, np.nan, np.median(ap_mad_chunks, axis=0)),
        car_mad_uv=np.where(flat, np.nan, np.median(car_mad_chunks, axis=0)),
        spike_rate_hz=np.where(flat, np.nan, spike_rate), spike_amp_uv=spike_amp,
        peak_time_s=peak_t, peak_channel=peak_ch, peak_amp_uv=peak_amp,
        artifact_bin_start_s=np.flatnonzero(artifact) * bin_s,
        freqs=freqs, psd=psd, band_names=np.array(band_names), band_power_db=band_power_db,
        row_ypos=row_ypos, psd_rows=psd_rows, corr=corr, corr_car=corr_car,
    )


def save_depth_profile(profile, path):
    """Save a profile from compute_depth_profile to .npz (params stored as JSON)."""
    out = {k: v for k, v in profile.items() if k != "params"}
    out["params_json"] = json.dumps(profile["params"])
    np.savez_compressed(path, **out)


def load_depth_profile(path):
    with np.load(path, allow_pickle=False) as z:
        profile = {k: z[k] for k in z.files}
    profile["params"] = json.loads(str(profile.pop("params_json")))
    for k in ("recording", "probe", "sample_rate", "duration_s", "clean_duration_s"):
        if k in profile:                            # "probe" is missing in caches made before it was added
            profile[k] = profile[k].item()
    return profile


def ypos_to_depth(ypos, implant_depth_um, tip_length_um):
    """Depth below the brain surface (um) of an electrode at `ypos` (um above the lowest row)."""
    return implant_depth_um - tip_length_um - np.asarray(ypos, dtype=float)


def _edges(centers):
    centers = np.asarray(centers, dtype=float)
    mid = (centers[1:] + centers[:-1]) / 2
    return np.concatenate([[centers[0] - (mid[0] - centers[0])], mid, [centers[-1] + (centers[-1] - mid[-1])]])


def _running_median(depth, values, depth_out, half_window_um):
    return np.array([np.nanmedian(values[np.abs(depth - d) <= half_window_um])
                     if np.any(np.isfinite(values[np.abs(depth - d) <= half_window_um])) else np.nan
                     for d in depth_out])


def plot_depth_profile(profile, implant_depth_um, tip_length_um, smooth_um=100, max_freq_hz=300,
                       title=None):
    """
    Seven panels of one shank against depth below the brain surface (surface = 0, black line):
    AP noise, threshold-crossing rate, raster, LFP band power, relative LFP spectrum, and the LFP
    depth-depth correlation without / with CAR. The right axis of the first panel gives ypos.
    """
    to_depth = lambda y: ypos_to_depth(y, implant_depth_um, tip_length_um)
    depth = to_depth(profile["ypos"])
    flat = profile["flat"]
    row_depth = to_depth(profile["row_ypos"])
    ylim = (depth.max() + 60, depth.min() - 60)

    fig = plt.figure(figsize=(26, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1.4, 1.2, 1.4, 2.6, 2.6])
    ax = [fig.add_subplot(gs[0])]
    ax += [fig.add_subplot(gs[i], sharey=ax[0]) for i in range(1, 5)]
    ax += [fig.add_subplot(gs[5]), fig.add_subplot(gs[6])]

    # 1. AP noise
    ax[0].scatter(profile["ap_mad_uv"], depth, s=5, c="tab:blue")
    ax[0].scatter(np.zeros(flat.sum()), depth[flat], s=15, marker="x", c="0.5", label="flat channel")
    ax[0].set_xlabel("AP noise, MAD (uV)")
    ax[0].set_ylabel("depth below brain surface (um)")
    ax[0].set_title("AP-band noise\n(300-6000 Hz, no CAR)")
    if flat.any():
        ax[0].legend(loc="lower right", fontsize=7)
    sec = ax[0].secondary_yaxis(-0.45, functions=(lambda d: implant_depth_um - tip_length_um - d,
                                                  lambda y: implant_depth_um - tip_length_um - y))
    sec.set_ylabel("ypos (um above lowest electrode row)")

    # 2. threshold-crossing rate
    rate = profile["spike_rate_hz"]
    ax[1].scatter(rate, depth, s=5, c="0.6")
    ax[1].plot(_running_median(depth, rate, row_depth, smooth_um / 2), row_depth, c="tab:red", lw=1.5,
               label=f"running median ({smooth_um} um)")
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].set_xlim(left=0)
    ax[1].set_xlabel("rate (Hz)")
    ax[1].set_title(f"threshold crossings\n(-{profile['params']['threshold_mad']:g} MAD, CAR)")
    ax[1].legend(loc="lower right", fontsize=7)

    # 3. raster
    t, ch, amp = profile["peak_time_s"], profile["peak_channel"], profile["peak_amp_uv"]
    sc = ax[2].scatter(t, depth[ch], s=0.4, c=np.clip(amp, 0, 150), cmap="viridis", rasterized=True,
                       linewidths=0)
    for t0 in profile["artifact_bin_start_s"]:
        ax[2].axvspan(t0, t0 + profile["params"]["artifact_bin_ms"] * 1e-3, color="red", alpha=0.3, lw=0)
    ax[2].set_xlim(0, profile["duration_s"])
    ax[2].set_xlabel("time in window (s)")
    ax[2].set_title(f"crossings raster\n(red = removed artifact bins: {len(profile['artifact_bin_start_s'])})")
    fig.colorbar(sc, ax=ax[2], location="bottom", shrink=0.8, label="amplitude (uV, clipped at 150)")

    # 4. LFP band power, mean per electrode row
    for name, bp in zip(profile["band_names"], profile["band_power_db"]):
        row_bp = [np.nanmean(bp[profile["ypos"] == y]) for y in profile["row_ypos"]]
        ax[3].plot(row_bp, row_depth, lw=1, label=str(name))
    ax[3].set_xlabel("power (dB, uV^2/Hz)")
    ax[3].set_title("LFP band power")
    ax[3].legend(loc="lower right", fontsize=7)

    # 5. LFP spectrum relative to the median over depth at each frequency
    f = profile["freqs"]
    sel = (f >= 0.5) & (f <= max_freq_hz)
    psd_db = 10 * np.log10(profile["psd_rows"][:, sel])
    rel = psd_db - np.median(psd_db, axis=0)
    pm = ax[4].pcolormesh(_edges(f[sel]), _edges(row_depth[::-1]), rel[::-1], cmap="RdBu_r",
                          vmin=-6, vmax=6, rasterized=True)
    ax[4].set_xscale("log")
    ax[4].set_xlabel("frequency (Hz)")
    ax[4].set_title("LFP spectrum, relative to\nmedian over depth (dB)")
    fig.colorbar(pm, ax=ax[4], location="bottom", shrink=0.8, label="dB re. median over depth")

    ax[0].set_ylim(*ylim)
    for a in ax[:5]:
        a.axhline(0, c="k", lw=0.8)

    # 6-7. LFP depth-depth correlation
    e = _edges(row_depth[::-1])
    for a, key, label in ((ax[5], "corr", "no CAR"), (ax[6], "corr_car", "CAR")):
        m = a.pcolormesh(e, e, profile[key][::-1, ::-1], cmap="RdBu_r", vmin=-1, vmax=1, rasterized=True)
        a.set_aspect("equal")
        a.set_xlim(ylim[1], ylim[0])
        a.set_ylim(*ylim)
        a.axhline(0, c="k", lw=0.8)
        a.axvline(0, c="k", lw=0.8)
        lo_f, hi_f = profile["params"]["corr_band_hz"]
        a.set_title(f"LFP correlation between depths\n({lo_f}-{hi_f} Hz, {label})")
        a.set_xlabel("depth (um)")
        fig.colorbar(m, ax=a, location="bottom", shrink=0.6, label="Pearson r")
    ax[5].set_ylabel("depth (um)")

    fig.suptitle(title or profile["recording"], fontsize=13)
    return fig


def plot_shank_overview(profiles, shank_ap_mm, implant_depth_um, tip_length_um, bin_um=50,
                        shank_width_mm=0.2):
    """
    Heatmaps across shanks: threshold-crossing rate and each LFP band power, binned in depth
    (`bin_um`) and placed at each shank's AP coordinate. White = no recorded channel in the bin.

    profiles    : list of profiles (any order; one shank each)
    shank_ap_mm : {shank index: AP coordinate in mm}
    """
    all_depth = np.concatenate([ypos_to_depth(p["ypos"], implant_depth_um, tip_length_um) for p in profiles])
    bins = np.arange(np.floor(all_depth.min() / bin_um) * bin_um, all_depth.max() + bin_um, bin_um)

    layers = [("threshold-crossing rate\nlog10(Hz + 0.1)",
               lambda p: np.log10(p["spike_rate_hz"] + 0.1), "viridis")]
    for i, name in enumerate(profiles[0]["band_names"]):
        layers.append((f"LFP {name}\n(dB)", lambda p, i=i: p["band_power_db"][i], "magma"))

    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 9), sharey=True,
                             constrained_layout=True)
    for ax, (label, getter, cmap) in zip(axes, layers):
        grids = []
        for p in profiles:
            depth = ypos_to_depth(p["ypos"], implant_depth_um, tip_length_um)
            v = getter(p)
            idx = np.digitize(depth, bins) - 1
            grids.append([np.nanmean(v[idx == k]) if np.any(np.isfinite(v[idx == k])) else np.nan
                          for k in range(len(bins) - 1)])
        vmin, vmax = np.nanpercentile(np.concatenate(grids), [2, 98])
        for p, g in zip(profiles, grids):
            ap = shank_ap_mm[int(p["shank"][0])]
            m = ax.pcolormesh([ap - shank_width_mm / 2, ap + shank_width_mm / 2], bins,
                              np.array(g)[:, None], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(sorted(shank_ap_mm.values()))
        ax.set_xticklabels([f"{v:.3f}\nsh{k}" for k, v in sorted(shank_ap_mm.items(), key=lambda kv: kv[1])],
                           fontsize=8)
        ax.set_xlabel("AP (mm)")
        ax.axhline(0, c="k", lw=0.8)
        ax.set_title(label, fontsize=10)
        fig.colorbar(m, ax=ax, location="bottom", shrink=0.8)
    axes[0].set_ylim(bins[-1], bins[0])
    axes[0].set_ylabel("depth below brain surface (um)")
    return fig



SITE_UM = 12.0          # NP2.0 electrode site, 12 x 12 um
SHANK_WIDTH_UM = 70.0
SHANK_PITCH_UM = 250.0


def channel_metrics(profile, large_amp_uv=100.0, boundary_span_um=100.0):
    """
    Per-channel metrics of one profile, for plot_probe_metric:
    {name: (values (n_ch,), colorbar label)}. Flat channels are NaN.

    The LFP boundary index of a channel is 1 - r between the CAR LFP of the electrode rows
    `boundary_span_um` below and above it; high where the LFP changes across depth.
    """
    flat = profile["flat"]
    n_ch = len(flat)
    clean = profile["clean_duration_s"]
    large = np.bincount(profile["peak_channel"][profile["peak_amp_uv"] > large_amp_uv], minlength=n_ch) / clean

    row_ypos = profile["row_ypos"]
    row_boundary = np.full(len(row_ypos), np.nan)
    for i, y in enumerate(row_ypos):
        j = np.argmin(np.abs(row_ypos - (y - boundary_span_um)))
        k = np.argmin(np.abs(row_ypos - (y + boundary_span_um)))
        if (abs(row_ypos[j] - (y - boundary_span_um)) <= boundary_span_um / 2
                and abs(row_ypos[k] - (y + boundary_span_um)) <= boundary_span_um / 2):
            row_boundary[i] = 1 - profile["corr_car"][j, k]
    boundary = np.full(n_ch, np.nan)
    for i, y in enumerate(row_ypos):
        boundary[(profile["ypos"] == y) & ~flat] = row_boundary[i]

    metrics = {
        "ap_noise": (profile["ap_mad_uv"], "AP-band noise, MAD (uV)"),
        "mua_rate": (np.log10(profile["spike_rate_hz"] + 0.1),
                     f"threshold-crossing rate, log10(Hz + 0.1)  (-{profile['params']['threshold_mad']:g} MAD)"),
        "large_spike_rate": (np.where(flat, np.nan, np.log10(large + 0.01)),
                             f"rate of crossings > {large_amp_uv:g} uV, log10(Hz + 0.01)"),
        "spike_amplitude": (profile["spike_amp_uv"], "median crossing amplitude (uV)"),
    }
    for name, bp in zip(profile["band_names"], profile["band_power_db"]):
        key = "lfp_" + str(name).rsplit(" ", 2)[0].replace(" ", "_")     # e.g. lfp_high_gamma
        metrics[key] = (bp, f"LFP power {name} (dB)")
    metrics["lfp_boundary_index"] = (boundary, f"LFP boundary index, 1 - r(CAR LFP {boundary_span_um:g} um "
                                               "above vs below)")
    return metrics


def plot_probe_metric(profiles, metric, shank_ap_mm, implant_depth_um, tip_length_um,
                      anterior_left=False, fill=True, max_fill_um=50.0, site_um=SITE_UM, cmap="turbo",
                      percentiles=(2, 98), large_amp_uv=100.0, boundary_span_um=100.0, inch_per_mm=2.0):
    """
    One metric drawn on the probe at true scale (1 mm horizontal = 1 mm vertical), every channel
    at its real position, for overlay on a sagittal atlas section (the shanks lie along A-P).
    x = AP (mm), y = depth below the brain surface (mm); shank outlines with the tip, dashed line =
    expected brain surface, grey = flat channel.

    Each shank has its own colour scale (`percentiles` of its values) and colour bar, because the
    shanks were recorded at different times.

    fill          : True = each channel fills its half of the shank (its electrode column) and
                    extends up / down to the midpoint with the neighbouring channel of the same
                    column (at most `max_fill_um` per side); False = site_um squares at the sites
    anterior_left : False = AP increases from left to right; True puts anterior on the left
    inch_per_mm   : fixed drawing scale; axes position and size are fixed (no automatic layout),
                    so every metric's figure has exactly the same geometry
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon, Rectangle

    cmap = plt.get_cmap(cmap).with_extremes(bad="0.6")
    to_depth = lambda y: ypos_to_depth(y, implant_depth_um, tip_length_um) / 1000
    tip = implant_depth_um / 1000
    top_depth = min(to_depth(p["ypos"].max()) for p in profiles) - 0.1
    half_w = SHANK_WIDTH_UM / 2000

    def to_ap(shank, x_um):          # shank AP + offset across the shank width (mm)
        center = shank * SHANK_PITCH_UM + 43.0      # electrode columns at 27 / 59 um (+ shank * pitch)
        return shank_ap_mm[shank] + (x_um - center) / 1000

    width_mm = (max(shank_ap_mm.values()) - min(shank_ap_mm.values())) + 0.6
    height_mm = tip + 0.2 - top_depth
    left, right, bottom, top = 1.0, 1.6, 0.8, 1.2           # margins (inch)
    ax_w, ax_h = width_mm * inch_per_mm, height_mm * inch_per_mm
    fig_w, fig_h = left + ax_w + right, bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])

    outlines, label = [], None
    ordered = sorted(profiles, key=lambda p: int(p["shank"][0]))
    for n, p in enumerate(ordered):
        shank = int(p["shank"][0])
        values, label = channel_metrics(p, large_amp_uv, boundary_span_um)[metric]
        sites, colors = [], []
        if fill:
            side = np.array([to_ap(shank, x) > shank_ap_mm[shank] for x in p["xpos"]])
            for half in (False, True):                      # each half of the shank = one electrode column
                idx = np.flatnonzero(side == half)
                idx = idx[np.argsort(p["ypos"][idx])]
                y = p["ypos"][idx]
                gaps = np.diff(y)
                below = np.minimum(np.concatenate([[gaps[0] if len(gaps) else site_um], gaps]) / 2, max_fill_um)
                above = np.minimum(np.concatenate([gaps, [gaps[-1] if len(gaps) else site_um]]) / 2, max_fill_um)
                x0 = shank_ap_mm[shank] if half else shank_ap_mm[shank] - half_w
                for i, yy, lo, hi in zip(idx, y, below, above):
                    # higher ypos = shallower, so the tile's top edge is at ypos + hi
                    sites.append(Rectangle((x0, to_depth(yy + hi)), half_w, (lo + hi) / 1000))
                    colors.append(values[i])
        else:
            for x, y, v in zip(p["xpos"], p["ypos"], values):
                ap, d = to_ap(shank, x), to_depth(y)
                sites.append(Rectangle((ap - site_um / 2000, d - site_um / 2000), site_um / 1000, site_um / 1000))
                colors.append(v)

        c = shank_ap_mm[shank]
        bottom_row = to_depth(0) + site_um / 2000 + 0.01
        outline_xy = [(c - half_w, top_depth), (c + half_w, top_depth), (c + half_w, bottom_row), (c, tip),
                      (c - half_w, bottom_row)]
        outlines.append(Polygon(outline_xy, closed=True))
        ax.add_patch(Polygon(outline_xy, closed=True, facecolor="0.93", edgecolor="none"))

        vmin, vmax = np.nanpercentile(values, percentiles)
        pc = PatchCollection(sites, cmap=cmap, edgecolor="none")
        pc.set_array(np.ma.masked_invalid(np.array(colors, dtype=float)))
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)

        # one colour bar per shank, stacked in the right margin (shank 0 on top)
        bar_h = ax_h * 0.12
        y_bar = bottom + ax_h * 0.66 - n * (bar_h + 0.55)    # below the brain-surface label
        cax = fig.add_axes([(left + ax_w + 0.25) / fig_w, y_bar / fig_h, 0.18 / fig_w, bar_h / fig_h])
        fig.colorbar(pc, cax=cax)
        cax.set_title(f"sh{shank}", fontsize=8)
        cax.tick_params(labelsize=7)

    ax.add_collection(PatchCollection(outlines, facecolor="none", edgecolor="0.3", lw=0.6))
    ax.axhline(0, c="k", ls="--", lw=0.8)
    ax.annotate("expected\nbrain surface", xy=(1.02, 0), xycoords=("axes fraction", "data"), fontsize=7,
                va="center", annotation_clip=False)

    ax.set_aspect("equal")
    ax.set_xlim(min(shank_ap_mm.values()) - 0.3, max(shank_ap_mm.values()) + 0.3)
    if anterior_left:
        ax.invert_xaxis()
    ax.set_ylim(tip + 0.2, top_depth)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.set_xlabel("AP from bregma (mm)" + ("   <- anterior" if anterior_left else "   anterior ->"))
    ax.set_ylabel("depth below brain surface (mm)")
    secx = ax.secondary_xaxis("top")
    secx.set_xticks(sorted(shank_ap_mm.values()))
    secx.set_xticklabels([f"sh{s}" for s, _ in sorted(shank_ap_mm.items(), key=lambda kv: kv[1])], fontsize=8)
    fig.text(0.02, 1 - 0.15 / fig_h, f"{metric}\n" + textwrap.fill(label, 60)
             + f"\ncolour scale per shank ({percentiles[0]:g}-{percentiles[1]:g} percentile)",
             fontsize=9, va="top")
    fig.text(0.02, 0.1 / fig_h, f"scale: 1 mm = {inch_per_mm:g} inch", fontsize=7)
    return fig
