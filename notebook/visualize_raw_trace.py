# This script is used to visualize the recording channel map and recorded ephys data (raw / processed data)


#%%
# path and params
from pathlib import Path
base_path = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-18_17-19-35_manualsurvey_shank1_bankAB")
# time range is relative to the start of ephys recording, instead of the time in timestamp
# T_START = 26.8
# T_END   = 27.8
# T_START = 30.4
# T_END   = 31.4
T_START = 111.8
T_END   = 112.8
save_dir = Path(r"C:\Users\Jeff\Downloads\output_optonpx")


#%%
import sys
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.getipython import get_ipython
from pprint import pprint
sys.path.append(str(Path(__file__).parent.parent) if "__file__" in dir() else "..")
ipython = get_ipython()
if ipython is not None:
    print(f"This script is running in an IPython environment.")
# basic recording info
from src import oe_parse_folders, oe_parse_params
oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, adc_params = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])


#%%
# visualize channel map
from src import plot_electrode_map
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")
    ipython.run_line_magic("config", "InlineBackend.figure_format = 'svg'")
probe = probes_params[0]
fig = plot_electrode_map(probe["channel_electrode"], title=base_path.name)
save_dir.mkdir(exist_ok=True)
if ipython is not None:
    plt.show()
fig.savefig(save_dir / f"{base_path.name}_channel_map.svg", bbox_inches="tight")


#%%
# load raw data slices with padding
from src import oe_load_ephys_slice
probe_dir = oe_paths["ephys_streams"][0]
raw = oe_load_ephys_slice(probe_dir, probes_params[0],
                          t_start=T_START,
                          t_end=T_END,
                          pad_s=0.1,
                          is_relative_time=True)
print(f"data shape : {raw.data.shape}  (n_channels × n_samples_with_pad)")
print(f"t range    : {raw.t_arr[0]:.4f} – {raw.t_arr[-1]:.4f} s  (OE clock)")
print(f"valid_slice: {raw.valid_slice}  → {raw.data[:, raw.valid_slice].shape[1]} valid samples")
print(f"µV range   : {raw.data.min():.2f} – {raw.data.max():.2f} µV")


#%%
# prepare colors for all channels
from src import find_contiguous_channel_groups
probe = probes_params[0]
groups = find_contiguous_channel_groups(probe)
for shank_idx, runs in enumerate(groups):
    if not runs:
        print(f"Shank {shank_idx}: no channels")
    else:
        for r, run in enumerate(runs):
            ypos_vals = sorted(probe["channel_ypos"][ch] for ch in run)
            print(f"Shank {shank_idx}  run {r}: {len(run):3d} channels, "
                  f"y {ypos_vals[0]:5d}–{ypos_vals[-1]:5d} µm")

import matplotlib.colors as mcolors
def make_channel_colors(channels):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "wide_gradient",
        ["#6a1b9a", "#1565c0", "#0277bd", "#00838f",
         "#2e7d32", "#558b2f", "#f57f17", "#e65100", "#b71c1c"],
    )
    norm = mcolors.Normalize(vmin=min(channels), vmax=max(channels))
    return {ch: cmap(norm(ch)) for ch in channels}
channels = sorted(probe["channel_electrode"].keys())
channel_color = make_channel_colors(channels)

special_channels = {
    "dead":  [],   # e.g. [72, 203]
    "noise": [],   # e.g. [154]
    "out":   [],
}


#%%
# plot function for all the traces
def plot_ephys_traces(data, t, groups, probe_params, channel_color,
                      special_channels=None, offset_uv=200, title_suffix="",
                      recording_name=""):
    label_style = {
        "dead":  {"color": "#cccccc", "lw": 0.3, "alpha": 0.8, "suffix": " [dead]"},
        "noise": {"color": "#bbbbbb", "lw": 0.3, "alpha": 0.8, "suffix": " [noise]"},
        "out":   {"color": "#666666", "lw": 0.4, "alpha": 0.9, "suffix": " [out]"},
    }

    ch_label = {}
    if special_channels:
        for label, chs in special_channels.items():
            for ch in chs:
                ch_label[ch] = label

    channel_ypos = probe_params["channel_ypos"]
    figs = []

    for shank_idx, runs in enumerate(groups):
        for run_idx, run_chs in enumerate(runs):
            if not run_chs:
                continue

            run_chs_sorted = sorted(run_chs, key=lambda ch: channel_ypos[ch])
            n_ch = len(run_chs_sorted)

            fig, ax = plt.subplots(figsize=(14, min(max(3, n_ch * 0.15), 20)))

            for i, ch in enumerate(run_chs_sorted):
                style = label_style.get(ch_label.get(ch, ""))
                color = style["color"] if style else channel_color[ch]
                lw    = style["lw"]    if style else 0.4
                alpha = style["alpha"] if style else 1.0
                ax.plot(t, data[ch] + i * offset_uv,
                        lw=lw, color=color, alpha=alpha)

            tick_pos = [i * offset_uv for i in range(n_ch)]
            ax.set_yticks(tick_pos)
            ax.tick_params(axis="y", labelleft=False, length=3)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Channel")
            ax.set_title(
                f"Shank {shank_idx}  run {run_idx}  ({n_ch} ch, "
                f"y {min(channel_ypos[ch] for ch in run_chs_sorted)}–"
                f"{max(channel_ypos[ch] for ch in run_chs_sorted)} µm)"
                + (f"  [{title_suffix}]" if title_suffix else "")
            )
            ax.set_xlim(t[0], t[-1])
            if recording_name:
                fig.suptitle(recording_name, fontsize=9, color="#555555", y=1.002)
            fig.tight_layout()
            figs.append((fig, shank_idx, run_idx))

    return figs


#%%
# plot the raw trace (always in an independent window)
if ipython is not None:
    ipython.run_line_magic("matplotlib", "qt")
data_valid = raw.data[:, raw.valid_slice]
t_valid    = raw.t_arr[raw.valid_slice]
data_plot  = data_valid - np.median(data_valid, axis=1, keepdims=True)
figs = plot_ephys_traces(data_plot, t_valid, groups, probe, channel_color,
                         special_channels=special_channels, offset_uv=400, title_suffix="raw",
                         recording_name=base_path.name)
for fig, sh, run in figs:
    if ipython is not None:
        plt.show()
    fig.savefig(save_dir / f"{base_path.name}_sh{sh}_run{run}_raw.svg", bbox_inches="tight")
    if ipython is None:
        plt.close(fig)


#%%
# plot filtered LFP data
if ipython is not None:
    ipython.run_line_magic("matplotlib", "qt")
from scipy.signal import butter, sosfiltfilt
LOWPASS_HZ = 300.0
sos_lp     = butter(4, LOWPASS_HZ, btype="lowpass", fs=raw.sample_rate, output="sos")
data_lfp   = sosfiltfilt(sos_lp, raw.data, axis=1).astype("float32")[:, raw.valid_slice]
data_lfp  -= np.median(data_lfp, axis=1, keepdims=True)
figs = plot_ephys_traces(data_lfp, t_valid, groups, probe, channel_color,
                         special_channels=special_channels, offset_uv=500,
                         title_suffix=f"LFP, LP {LOWPASS_HZ:.0f} Hz",
                         recording_name=base_path.name)
for fig, sh, run in figs:
    if ipython is not None:
        plt.show()
    fig.savefig(save_dir / f"{base_path.name}_sh{sh}_run{run}_LFP_LP{LOWPASS_HZ:.0f}Hz.svg", bbox_inches="tight")
    if ipython is None:
        plt.close(fig)

#%%
# plot filtered and CARed spike data
if ipython is not None:
    ipython.run_line_magic("matplotlib", "qt")
from scipy.signal import butter, sosfiltfilt
HIGHPASS_HZ = 300.0
sos         = butter(4, HIGHPASS_HZ, btype="highpass", fs=raw.sample_rate, output="sos")
data_filt   = sosfiltfilt(sos, raw.data, axis=1).astype("float32")[:, raw.valid_slice]
data_car = data_filt.copy()
for runs in groups:
    for run_chs in runs:
        if not run_chs:
            continue
        ch_arr = np.array(run_chs)
        data_car[ch_arr] -= data_filt[ch_arr].mean(axis=0)
data_car -= np.median(data_car, axis=1, keepdims=True)
figs = plot_ephys_traces(data_car, t_valid, groups, probe, channel_color,
                         special_channels=special_channels, offset_uv=200,
                         title_suffix=f"HP {HIGHPASS_HZ:.0f} Hz + CAR",
                         recording_name=base_path.name)
for fig, sh, run in figs:
    if ipython is not None:
        plt.show()
    fig.savefig(save_dir / f"{base_path.name}_sh{sh}_run{run}_HP{HIGHPASS_HZ:.0f}Hz_CAR.svg", bbox_inches="tight")
    if ipython is None:
        plt.close(fig)
    plt.show()


# %%
