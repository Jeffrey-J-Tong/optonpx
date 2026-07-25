# Build a SpikeInterface SortingAnalyzer from Phy-curated Kilosort4 output.
# Reads spike_clusters.npy (merge/split) and cluster_group.tsv (manual labels)
# from the KS4 directory and saves a new analyzer.zarr with all standard extensions.
#
# Run AFTER Phy manual curation is complete.  Raises FileExistsError if
# analyzer.zarr already exists to prevent accidental overwrites.
#
# Environment: spikeinterface  (conda activate spikeinterface)

#%%
# ── paths and params ───────────────────────────────────────────────────────────
from pathlib import Path

base_path      = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14")
ks_output_name = "kilosort4_1_bombcell"
PROBE_IDX      = 0
SHANK_IDX      = None

N_JOBS = 1   # multiprocessing > 1 does not work on Windows with plain .py scripts
# N_JOBS = 4


#%%
# ── resolve paths ──────────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent) if "__file__" in dir() else "..")

from src import oe_parse_folders, oe_parse_params

oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, _   = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])

probe_params   = probes_params[PROBE_IDX]
stream_name    = probe_params["stream_name"]
sample_rate    = float(probe_params["sample_rate"])
n_chan_bin     = int(probe_params["channel_count"])
continuous_dir = oe_paths["ephys_streams"][PROBE_IDX]
dat_file       = continuous_dir / "continuous.dat"

ks_dir = continuous_dir / ks_output_name
if SHANK_IDX is not None:
    ks_dir = ks_dir / f"shank_{SHANK_IDX}"

analyzer_path = ks_dir / "analyzer.zarr"

if not ks_dir.exists():
    raise FileNotFoundError(f"KS4 output not found: {ks_dir}")
if not dat_file.exists():
    raise FileNotFoundError(f"Raw binary not found: {dat_file}")
if analyzer_path.exists():
    raise FileExistsError(
        f"analyzer.zarr already exists at {analyzer_path}\n"
        "Delete or rename it before running this script."
    )

print(f"Recording  : {base_path.name}")
print(f"Probe      : {stream_name}  ({n_chan_bin} ch,  {sample_rate} Hz)")
print(f"KS4 dir    : {ks_dir}")
print(f"Analyzer   : {analyzer_path}  (will be created)")


#%%
# ── load Phy-curated sorting ───────────────────────────────────────────────────
# se.read_kilosort reads spike_clusters.npy, which Phy modifies to reflect
# merge/split operations — so this gives the post-curation unit set.
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.core as si

sorting = se.read_kilosort(ks_dir, keep_good_only=False)
print(f"Loaded {len(sorting.unit_ids)} units (post-Phy merge/split)")

# Attach manual group labels (good / mua / noise) as a sorting property so
# they are saved into the zarr and visible in SpikeInterface.
cg_path = ks_dir / "cluster_group.tsv"
if not cg_path.exists():
    raise FileNotFoundError(f"cluster_group.tsv not found: {cg_path}  — run Phy first")

cg = pd.read_csv(cg_path, sep="\t")
cg_map = dict(zip(cg["cluster_id"].astype(int), cg["group"].str.lower().str.strip()))
group_labels = np.array(
    [cg_map.get(int(uid), "unsorted") for uid in sorting.unit_ids], dtype=str
)
sorting.set_property("group", group_labels)

from collections import Counter
print("Group label counts:", dict(Counter(group_labels)))


#%%
# ── load recording ─────────────────────────────────────────────────────────────
import spikeinterface.preprocessing as sp
from probeinterface import Probe

recording_raw = se.BinaryRecordingExtractor(
    file_paths=dat_file,
    sampling_frequency=sample_rate,
    num_channels=n_chan_bin,
    dtype="int16",
)
recording_filtered = sp.highpass_filter(recording_raw, freq_min=300.0)

chan_map = np.load(ks_dir / "channel_map.npy")       # (n_sorted_ch,)
chan_pos = np.load(ks_dir / "channel_positions.npy") # (n_sorted_ch, 2)

recording = recording_filtered.select_channels(
    channel_ids=[recording_filtered.channel_ids[i] for i in chan_map]
)
probe = Probe(ndim=2, si_units="um")
probe.set_contacts(positions=chan_pos, shapes="circle", shape_params={"radius": 5})
probe.set_device_channel_indices(np.arange(len(chan_map)))
recording = recording.set_probe(probe)

print(f"Recording  : {recording.get_num_channels()} ch  ×  "
      f"{recording.get_num_samples()} samples  ({recording.get_total_duration():.1f} s)")


#%%
# ── build sparsity from KS4 templates ─────────────────────────────────────────
# Derives the main channel per post-Phy unit from KS4 templates.npy using a
# weighted PTP average (weighted by how many spikes came from each KS4 template).
# Handles merges (multiple templates → weighted average) and splits (both child
# units get the same channels — correct since they share a spatial location).
# Then uses estimate_sparsity(method="radius") with those main channels so the
# final channel set follows probe geometry rather than a fixed top-N count.
from spikeinterface.core import ChannelSparsity

ks4_templates   = np.load(ks_dir / "templates.npy")       # (n_ks4_units, T, n_ch)
spike_clusters  = np.load(ks_dir / "spike_clusters.npy")  # post-Phy cluster per spike
spike_templates = np.load(ks_dir / "spike_templates.npy") # original KS4 template per spike

ptp_ks4  = ks4_templates.max(axis=1) - ks4_templates.min(axis=1)  # (n_ks4_units, n_ch)
unit_ids = sorting.unit_ids

# Build radius-based sparsity manually using probe geometry.
# For each unit: find main channel via weighted PTP, then include all channels
# within radius_um using Euclidean distance on channel positions.
RADIUS_UM = 100.0
mask = np.zeros((len(unit_ids), len(chan_map)), dtype=bool)
for i, uid in enumerate(unit_ids):
    spike_idx    = np.where(spike_clusters == int(uid))[0]
    tmpl_ids, tmpl_counts = np.unique(spike_templates[spike_idx], return_counts=True)
    weighted_ptp = np.average(ptp_ks4[tmpl_ids], axis=0, weights=tmpl_counts)
    main_ch = int(np.argmax(weighted_ptp))
    dists = np.linalg.norm(chan_pos - chan_pos[main_ch], axis=1)
    mask[i] = dists <= RADIUS_UM

sparsity = ChannelSparsity(mask, unit_ids, recording.channel_ids)
print(f"Sparsity built  (avg {mask.sum(axis=1).mean():.1f} ch/unit, radius={RADIUS_UM} µm)")


#%%
# ── create SortingAnalyzer and compute all extensions ─────────────────────────
print("Creating SortingAnalyzer ...")
analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording,
    format="zarr",
    folder=analyzer_path,
    sparse=True,
    sparsity=sparsity,
    overwrite=False,
)
print(f"Analyzer created: {analyzer_path}")

print("Computing extensions ...")
analyzer.compute("random_spikes",    method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms",        n_jobs=N_JOBS)
analyzer.compute("templates",        operators=["average", "std"])
analyzer.compute("noise_levels")
analyzer.compute("spike_amplitudes", n_jobs=N_JOBS)
analyzer.compute("spike_locations",  n_jobs=N_JOBS)
analyzer.compute("correlograms",     window_ms=100.0, bin_ms=1.0)
analyzer.compute("isi_histograms")
analyzer.compute("template_metrics")
analyzer.compute("quality_metrics",  n_jobs=N_JOBS)

print(f"\nDone. Extensions saved: {analyzer.get_saved_extension_names()}")


#%%
# ── verify ─────────────────────────────────────────────────────────────────────
analyzer2 = si.load_sorting_analyzer(analyzer_path)
print(f"Reloaded analyzer: {len(analyzer2.unit_ids)} units")
print(f"Extensions       : {analyzer2.get_saved_extension_names()}")

props = analyzer2.sorting.get_property_keys()
print(f"Sorting properties: {props}")
if "group" in props:
    grp = analyzer2.sorting.get_property("group")
    print("Group counts:", dict(Counter(grp)))
else:
    print("WARNING: 'group' property not found in zarr")
