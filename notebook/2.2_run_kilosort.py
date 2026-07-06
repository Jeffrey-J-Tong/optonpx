# Run Kilosort 4 on an Open Ephys recording.
# Prerequisites: generate_channel_map.py must have been run first to create the .mat probe file.
# Results are saved to results_dir (default: continuous_dir / 'kilosort4').


#%%
# ── paths ──────────────────────────────────────────────────────────────────────
from pathlib import Path

base_path   = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14")
PROBE_IDX   = 0        # index into probes_params for multi-probe recordings
results_dir = None     # None → continuous_dir / 'kilosort4'

# ── data ───────────────────────────────────────────────────────────────────────
tmin        = 0        # (s) start of data to sort; 0 = beginning of recording
tmax        = None     # (s) end   of data to sort; None = end of recording
batch_size  = 60000   # samples per batch (default 60000 = 2 s); reduce if GPU OOM

# ── bad channel exclusion ──────────────────────────────────────────────────────
# Bad channels are normally excluded via generate_channel_map.py → chanMap in the .mat file.
# Only set this if you need to exclude additional channels without regenerating the .mat,
# or if the channel map was generated without excluding them.
# Use the same 0-indexed channel numbers as in generate_channel_map.py.
bad_channels    = None     # e.g. [154] — list of 0-indexed channel indices

# ── shank selection ────────────────────────────────────────────────────────────
# None = sort all shanks together in one pass
# list of kcoords values (0-indexed) = sort each shank separately → results in subfolders
# e.g. [0, 1, 2, 3] for 4-shank NP2.0
shank_idx   = None

# ── preprocessing ──────────────────────────────────────────────────────────────
nblocks            = 1      # drift correction blocks: 1 = rigid, 0 = none, >1 = non-rigid
highpass_cutoff    = 300.0  # (Hz) highpass filter cutoff
do_CAR             = True   # common average reference during preprocessing (recommended)
artifact_threshold = None   # zero out batches with abs values above this; None = disabled
save_preprocessed_copy = False  # save temp_wh.dat for Phy; needed if you want Phy to show waveforms

# ── spike detection ────────────────────────────────────────────────────────────
Th_universal  = 9.0 #9.0   # detection threshold for universal templates (higher = stricter)
Th_learned    = 8.0 #8.0   # detection threshold for learned templates
Th_single_ch  = 6.0   # threshold for single-channel crossing used to fit universal templates
dminx         = 32.0  # (µm) horizontal spacing of template centers — 32 µm for NP2.0

# ── clustering ─────────────────────────────────────────────────────────────────
# None = KS4 determines automatically (may work poorly for multi-shank 2D arrays)
# set to number of shanks for NP2.0 multi-shank, e.g. 4
x_centers     = None


#%%
# ── resolve OE paths ───────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent) if "__file__" in dir() else "..")

from src import oe_parse_folders, oe_parse_params

oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, _   = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])

probe_params   = probes_params[PROBE_IDX]
stream_name    = probe_params["stream_name"]
n_chan_bin     = int(probe_params["channel_count"])
sample_rate    = float(probe_params["sample_rate"])
continuous_dir = oe_paths["ephys_streams"][PROBE_IDX]
dat_file       = continuous_dir / "continuous.dat"
probe_mat      = oe_paths["oebin"].parent / f"{base_path.name}_{stream_name}.mat"

if not dat_file.exists():
    raise FileNotFoundError(f"Binary file not found: {dat_file}")
if not probe_mat.exists():
    raise FileNotFoundError(f"Probe .mat not found: {probe_mat}  — run generate_channel_map.py first")

out_dir = Path(results_dir) if results_dir else continuous_dir / "kilosort4"

print(f"Recording  : {base_path.name}")
print(f"Probe      : {stream_name}  ({n_chan_bin} ch,  {sample_rate} Hz)")
print(f"Binary     : {dat_file}")
print(f"Probe map  : {probe_mat.name}")
print(f"Results dir: {out_dir}")


#%%
# ── run Kilosort 4 ─────────────────────────────────────────────────────────────
import numpy as np
from kilosort import run_kilosort
from kilosort.io import load_probe

probe = load_probe(probe_mat)

settings = {
    "filename":    dat_file,
    "n_chan_bin":  n_chan_bin,
    "fs":          sample_rate,
    "results_dir": out_dir,
    # data
    "tmin":       tmin,
    "tmax":       tmax if tmax is not None else np.inf,
    "batch_size": batch_size,
    # preprocessing
    "nblocks":           nblocks,
    "highpass_cutoff":   highpass_cutoff,
    # spike detection
    "Th_universal": Th_universal,
    "Th_learned":   Th_learned,
    "Th_single_ch": Th_single_ch,
    "dminx":        dminx,
    # clustering
    "x_centers": x_centers,
}
if artifact_threshold is not None:
    settings["artifact_threshold"] = artifact_threshold

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings=settings, probe=probe,
                 do_CAR=do_CAR,
                 save_preprocessed_copy=save_preprocessed_copy,
                 bad_channels=bad_channels,
                 shank_idx=shank_idx)

print(f"\nDone.  {is_ref.sum()} good units / {len(is_ref)} total units")


#%%
# ── summary plots ──────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

camps      = pd.read_csv(out_dir / "cluster_Amplitude.tsv",  sep="\t")["Amplitude"].values
contam_pct = pd.read_csv(out_dir / "cluster_ContamPct.tsv",  sep="\t")["ContamPct"].values
chan_map    = np.load(out_dir / "channel_map.npy")
templates   = np.load(out_dir / "templates.npy")
chan_best   = chan_map[(templates**2).sum(axis=1).argmax(axis=-1)]
st_arr      = np.load(out_dir / "spike_times.npy")
clu_arr     = np.load(out_dir / "spike_clusters.npy")
firing_rates = np.unique(clu_arr, return_counts=True)[1] * ops["fs"] / st_arr.max()
dshift      = ops["dshift"]

gray = 0.5 * np.ones(3)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.suptitle(base_path.name, fontsize=10)

axes[0, 0].plot(np.arange(ops["Nbatches"]) * 2, dshift)
axes[0, 0].set(xlabel="time (s)", ylabel="drift (µm)", title="drift")

t1 = np.nonzero(st_arr > ops["fs"] * 5)[0]
if len(t1):
    t1 = t1[0]
    axes[0, 1].scatter(st_arr[:t1] / ops["fs"], chan_best[clu_arr[:t1]],
                       s=0.5, color="k", alpha=0.25)
axes[0, 1].set(xlabel="time (s)", ylabel="channel", title="spikes (first 5 s)",
               xlim=[0, 5], ylim=[chan_map.max(), 0])

axes[0, 2].hist(firing_rates, 20, color=gray)
axes[0, 2].set(xlabel="firing rate (Hz)", ylabel="# units")

axes[1, 0].hist(camps, 20, color=gray)
axes[1, 0].set(xlabel="amplitude", ylabel="# units")

axes[1, 1].hist(np.minimum(100, contam_pct), np.arange(0, 105, 5), color=gray)
axes[1, 1].axvline(10, color="k", ls="--")
axes[1, 1].set(xlabel="% contamination", ylabel="# units", title="< 10% = good")

is_good = contam_pct < 10.
axes[1, 2].scatter(firing_rates[~is_good], camps[~is_good], s=3, color="r", alpha=0.25, label="mua")
axes[1, 2].scatter(firing_rates[is_good],  camps[is_good],  s=3, color="b", alpha=0.25, label="good")
axes[1, 2].set(xlabel="firing rate (Hz)", ylabel="amplitude", xscale="log", yscale="log")
axes[1, 2].legend(fontsize=8)

fig.tight_layout()
plt.show()
