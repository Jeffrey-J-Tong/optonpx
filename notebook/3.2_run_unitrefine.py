# Run UnitRefine automatic unit curation on Kilosort4 output.
# Prerequisites: run_kilosort.py must have been run first.
# Uses SpikeInterface to load KS4 output and apply pre-trained UnitRefine classifiers.
# Results are saved to ks_dir / 'unitrefine_labels.tsv' (Phy-compatible TSV).
#
# Environment: spikeinterface  (conda activate spikeinterface)


#%%
# ── paths and params ───────────────────────────────────────────────────────────
from pathlib import Path

base_path  = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14")
ks_output_name = "kilosort4_1"
PROBE_IDX  = 0       # must match the PROBE_IDX used in run_kilosort.py
# if shank_idx was used in run_kilosort.py, set this to sort results into the
# correct subfolder, e.g. 0 → ks_dir / 'shank_0'; None = no subfolder
SHANK_IDX  = None

# ── UnitRefine classifier models ───────────────────────────────────────────────
# Pre-trained models from HuggingFace (downloaded automatically on first run).
# Set either to None to skip that classification stage.
NOISE_NEURAL_MODEL = "SpikeInterface/UnitRefine_noise_neural_classifier"
SUA_MUA_MODEL      = "SpikeInterface/UnitRefine_sua_mua_classifier"

# ── SortingAnalyzer settings ───────────────────────────────────────────────────
# Extensions required by UnitRefine. The analyzer is saved to ks_dir/analyzer.zarr
# so it can be reused without recomputing. Set OVERWRITE_ANALYZER=True to recompute.
OVERWRITE_ANALYZER  = False
N_JOBS              = 1    # multiprocessing (n_jobs > 1) does not work on Windows with plain .py scripts


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
    raise FileNotFoundError(f"Kilosort4 output not found: {ks_dir}  — run run_kilosort.py first")
if not dat_file.exists():
    raise FileNotFoundError(f"Raw binary not found: {dat_file}")

print(f"Recording  : {base_path.name}")
print(f"Probe      : {stream_name}  ({n_chan_bin} ch,  {sample_rate} Hz)")
print(f"KS4 dir    : {ks_dir}")
print(f"Binary     : {dat_file}")
print(f"Analyzer   : {analyzer_path}")


#%%
# ── load KS4 sorting and recording ─────────────────────────────────────────────
import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.core as si

# Load KS4 output — includes all units (good + MUA)
sorting = se.read_kilosort(ks_dir, keep_good_only=False)
print(f"Loaded {len(sorting.unit_ids)} units from KS4 output")

# Wrap the raw binary as a SpikeInterface recording.
# n_chan_bin is the total column count in the .dat file — must match the actual
# number of columns so the binary is parsed correctly (includes any bad channels
# that were excluded from sorting).
recording_raw = se.BinaryRecordingExtractor(
    file_paths=dat_file,
    sampling_frequency=sample_rate,
    num_channels=n_chan_bin,
    dtype="int16",
)

# Apply the same highpass filter KS4 used so waveform features are consistent.
recording_filtered = sp.highpass_filter(recording_raw, freq_min=300.0)

# Slice to the channels KS4 actually sorted and attach probe geometry.
# KS4 writes channel_map.npy (0-indexed channel indices) and
# channel_positions.npy (x/y in µm) to the results directory.
# create_sorting_analyzer requires a Probe to estimate spatial sparsity.
from probeinterface import Probe

chan_map   = np.load(ks_dir / "channel_map.npy")          # shape: (n_sorted_ch,)
chan_pos   = np.load(ks_dir / "channel_positions.npy")    # shape: (n_sorted_ch, 2)

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
# ── build or load SortingAnalyzer ──────────────────────────────────────────────
# The analyzer stores precomputed features (templates, quality metrics, PCA) that
# UnitRefine classifiers use. Computing these takes several minutes; the result is
# cached as analyzer.zarr so subsequent runs skip recomputation.

# All extensions that must be present for UnitRefine to run.
REQUIRED_EXTENSIONS = [
    "random_spikes", "waveforms", "templates", "noise_levels",
    "spike_amplitudes", "spike_locations", "correlograms", "isi_histograms",
    "principal_components", "template_metrics", "quality_metrics",
]

def _build_sparsity(sorting, recording, ks_dir, chan_map):
    """Derive channel sparsity from KS4 templates — avoids scanning the recording."""
    from spikeinterface.core import ChannelSparsity
    ks4_templates = np.load(ks_dir / "templates.npy")   # (n_units, T, n_channels)
    ptp = ks4_templates.max(axis=1) - ks4_templates.min(axis=1)
    n_best = 12
    mask = np.zeros((len(sorting.unit_ids), len(chan_map)), dtype=bool)
    for i in range(len(sorting.unit_ids)):
        mask[i, np.argsort(ptp[i])[-n_best:]] = True
    sparsity = ChannelSparsity(mask, sorting.unit_ids, recording.channel_ids)
    print(f"Sparsity from KS4 templates  (avg {mask.sum(axis=1).mean():.1f} ch/unit)")
    return sparsity

def _compute_extensions(analyzer, n_jobs):
    analyzer.compute("random_spikes",        method="uniform", max_spikes_per_unit=500)
    analyzer.compute("waveforms",            n_jobs=n_jobs)
    analyzer.compute("templates",            operators=["average", "std"])
    analyzer.compute("noise_levels")
    analyzer.compute("spike_amplitudes",     n_jobs=n_jobs)
    analyzer.compute("spike_locations",      n_jobs=n_jobs)   # required for drift metrics
    analyzer.compute("correlograms",         window_ms=100.0, bin_ms=1.0)
    analyzer.compute("isi_histograms")
    analyzer.compute("principal_components", n_components=5, mode="by_channel_local", n_jobs=n_jobs)
    analyzer.compute("template_metrics")                      # waveform shape metrics (peak_to_valley, half_width, etc.)
    analyzer.compute("quality_metrics",      n_jobs=n_jobs)   # compute all metrics

if analyzer_path.exists() and not OVERWRITE_ANALYZER:
    print(f"Loading existing analyzer from {analyzer_path.name} ...")
    analyzer = si.load_sorting_analyzer(analyzer_path)
    saved = set(analyzer.get_saved_extension_names())
    missing = [e for e in REQUIRED_EXTENSIONS if e not in saved]
    if missing:
        print(f"Incomplete analyzer — missing extensions: {missing}")
        print("Recomputing all extensions ...")
        _compute_extensions(analyzer, N_JOBS)
    else:
        print("All required extensions present — skipping recomputation.")
else:
    sparsity = _build_sparsity(sorting, recording, ks_dir, chan_map)
    print("Creating SortingAnalyzer — this may take several minutes ...")
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="zarr",
        folder=analyzer_path,
        sparse=True,
        sparsity=sparsity,
        overwrite=OVERWRITE_ANALYZER,
    )
    _compute_extensions(analyzer, N_JOBS)
    print(f"Analyzer saved to {analyzer_path}")

print(f"Extensions: {analyzer.get_saved_extension_names()}")


#%%
# ── run UnitRefine ─────────────────────────────────────────────────────────────
from spikeinterface.curation import unitrefine_label_units

print("Running UnitRefine classifiers ...")
labels_df = unitrefine_label_units(
    sorting_analyzer=analyzer,
    noise_neural_classifier=NOISE_NEURAL_MODEL,
    sua_mua_classifier=SUA_MUA_MODEL,
)

# labels_df: index=unit_id, columns=[unitrefine_label, unitrefine_probability]
print(f"\nUnitRefine results ({len(labels_df)} units):")
print(labels_df["unitrefine_label"].value_counts().to_string())


#%%
# ── save results as Phy-compatible TSV ────────────────────────────────────────
import pandas as pd

tsv_path = ks_dir / "cluster_unitrefine_label.tsv"

# Phy expects: cluster_id (int) + label column
out = labels_df.reset_index().rename(columns={"unit_id": "cluster_id"})
out.to_csv(tsv_path, sep="\t", index=False)
print(f"Saved labels → {tsv_path}")

# Also write a cluster_group.tsv using UnitRefine labels so Phy loads them
# as the default curation column:
#   sua  → good
#   mua  → mua
#   noise → noise
group_map = {"sua": "good", "mua": "mua", "noise": "noise"}
cluster_group_path = ks_dir / "cluster_group.tsv"
if not cluster_group_path.exists():
    group_df = out[["cluster_id"]].copy()
    group_df["group"] = out["unitrefine_label"].map(group_map).fillna("unsorted")
    group_df.to_csv(cluster_group_path, sep="\t", index=False)
    print(f"Saved cluster_group.tsv → {cluster_group_path}  (Phy default curation)")
else:
    print(f"cluster_group.tsv already exists — not overwritten (Phy may have curated it)")


#%%
# ── (optional) merge with Bombcell labels ─────────────────────────────────────
# If run_bombcell.py has been run, merge its unit-type labels with UnitRefine.
bc_tsv = ks_dir / "cluster_bc_unitType.tsv"
if bc_tsv.exists():
    bc_df = pd.read_csv(bc_tsv, sep="\t").rename(columns={"cluster_id": "cluster_id"})
    merged = out.merge(bc_df, on="cluster_id", how="left")
    merged_path = ks_dir / "cluster_labels_merged.tsv"
    merged.to_csv(merged_path, sep="\t", index=False)
    print(f"\nMerged Bombcell + UnitRefine labels → {merged_path}")
    print(merged[["cluster_id", "unitrefine_label", "bc_unitType"]].head(10).to_string(index=False))
else:
    print("\nBombcell labels not found — run run_bombcell.py to get cell-type classifications")
    print(f"(expected: {bc_tsv})")
