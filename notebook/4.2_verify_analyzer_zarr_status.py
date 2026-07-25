# Inspect a SortingAnalyzer zarr created after KS4 sorting + UnitRefine.
# Read-only: nothing is written or recomputed.
# Also checks whether Phy manual curation results are reflected in the zarr.
#
# Environment: spikeinterface  (conda activate spikeinterface)

#%%
# ── paths ──────────────────────────────────────────────────────────────────────
from pathlib import Path

ks_dir = Path(
    r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14"
    r"\Record Node 101\experiment1\recording1\continuous"
    r"\OneBox-100.ProbeA\kilosort4_1_bombcell"
)

analyzer_path = ks_dir / "analyzer.zarr"

if not analyzer_path.exists():
    raise FileNotFoundError(f"analyzer.zarr not found at {analyzer_path}")


#%%
# ── load analyzer (read-only) ──────────────────────────────────────────────────
import spikeinterface.core as si

analyzer = si.load_sorting_analyzer(analyzer_path)

print(f"Analyzer format : {analyzer.format}")
print(f"Number of units : {len(analyzer.unit_ids)}")
print(f"Sampling rate   : {analyzer.sampling_frequency} Hz")
print(f"Number of channels: {analyzer.get_num_channels()}")
print(f"\nSaved extensions:")
for ext in analyzer.get_saved_extension_names():
    print(f"  {ext}")


#%%
# ── inspect unit properties stored in the zarr ────────────────────────────────
sorting = analyzer.sorting

print("Unit properties stored in zarr sorting:")
props = sorting.get_property_keys()
if props:
    for key in props:
        vals = sorting.get_property(key)
        unique = set(vals) if vals is not None else set()
        print(f"  {key:30s}  unique values: {sorted(unique)[:10]}")
else:
    print("  (none)")


#%%
# ── check Phy manual curation files ───────────────────────────────────────────
# Phy saves manual curation to TSV files in ks_dir, NOT inside the zarr.
# This cell compares unit_ids in the zarr vs what Phy has curated.
import pandas as pd

cluster_group_tsv = ks_dir / "cluster_group.tsv"
cluster_info_tsv  = ks_dir / "cluster_info.tsv"

print("── Phy curation files ────────────────────────────────────────────────────")
if cluster_group_tsv.exists():
    cg = pd.read_csv(cluster_group_tsv, sep="\t")
    print(f"\ncluster_group.tsv  ({len(cg)} rows)")
    print(cg["group"].value_counts().to_string())
else:
    print("\ncluster_group.tsv  NOT FOUND — Phy may not have been run yet")

if cluster_info_tsv.exists():
    ci = pd.read_csv(cluster_info_tsv, sep="\t")
    print(f"\ncluster_info.tsv  ({len(ci)} rows)")
    if "group" in ci.columns:
        print("  'group' column (manual curation):")
        print("  " + ci["group"].value_counts().to_string().replace("\n", "\n  "))
    else:
        print("  No 'group' column found — curation may not be present")
else:
    print("\ncluster_info.tsv  NOT FOUND")


#%%
# ── compare zarr unit_ids vs Phy cluster_ids ──────────────────────────────────
print("── Overlap between zarr units and Phy curated clusters ──────────────────")

zarr_ids = set(int(uid) for uid in analyzer.unit_ids)

if cluster_info_tsv.exists():
    ci = pd.read_csv(cluster_info_tsv, sep="\t")
    phy_ids = set(ci["cluster_id"].astype(int))

    only_in_zarr = zarr_ids - phy_ids
    only_in_phy  = phy_ids  - zarr_ids
    in_both      = zarr_ids & phy_ids

    print(f"\n  Units in zarr              : {len(zarr_ids)}")
    print(f"  Clusters in cluster_info   : {len(phy_ids)}")
    print(f"  In both                    : {len(in_both)}")
    if only_in_zarr:
        print(f"  Only in zarr (not in Phy)  : {sorted(only_in_zarr)[:20]}")
    if only_in_phy:
        print(f"  Only in Phy  (not in zarr) : {sorted(only_in_phy)[:20]}")

    # Check if the zarr sorting already carries the manual 'group' label
    if "group" in sorting.get_property_keys():
        print("\n  ✓ 'group' property IS stored in the zarr sorting.")
        print("    The zarr reflects Phy manual curation.")
    else:
        print("\n  ✗ 'group' property is NOT stored in the zarr sorting.")
        print("    Phy curation only lives in the TSV files, not in the zarr.")
        print("    To use manual labels in analysis, read from cluster_info.tsv directly.")

# %%
