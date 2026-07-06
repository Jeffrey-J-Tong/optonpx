# Run Bombcell quality metrics and cell-type classification on Kilosort4 output.
# Prerequisites: run_kilosort.py must have been run first.
# Results are saved to ks_dir / 'bombcell/' and cluster TSVs to ks_dir/ (Phy-compatible).
#
# Environment: bombcell  (conda activate bombcell)


#%%
# ── paths and params ───────────────────────────────────────────────────────────
from pathlib import Path

base_path  = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14")
ks_output_name = "kilosort4_1"
PROBE_IDX  = 0       # must match the PROBE_IDX used in run_kilosort.py
# if shank_idx was used in run_kilosort.py, set this to the same shank index,
# e.g. 0 → ks_dir / 'shank_0'; None = no subfolder
SHANK_IDX  = None

# ── Bombcell parameters ────────────────────────────────────────────────────────
# Refractory period violation method: "hill" (recommended) or "llobet"
RPV_METHOD = "hill"

# ── cell-type classification ───────────────────────────────────────────────────
# Brain region determines which cell types are classified:
#   "cortex"   → Wide-spiking (>400 µs) / Narrow-spiking (≤400 µs)
#   "striatum" → MSN / FSI / TAN / UIN  (waveform duration + PSS + propLongISI)
BRAIN_REGION = "cortex"


#%%
# ── resolve paths from OE folder ──────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent) if "__file__" in dir() else "..")

from src import oe_parse_folders, oe_parse_params

oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, _   = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])

probe_params   = probes_params[PROBE_IDX]
stream_name    = probe_params["stream_name"]
continuous_dir = oe_paths["ephys_streams"][PROBE_IDX]
raw_file       = continuous_dir / "continuous.dat"
meta_file      = oe_paths["oebin"]

ks_dir = continuous_dir / ks_output_name
if SHANK_IDX is not None:
    ks_dir = ks_dir / f"shank_{SHANK_IDX}"

if not ks_dir.exists():
    raise FileNotFoundError(f"Kilosort4 output not found: {ks_dir}  — run run_kilosort.py first")
if not raw_file.exists():
    raise FileNotFoundError(f"Raw binary not found: {raw_file}")

print(f"Recording  : {base_path.name}")
print(f"Probe      : {stream_name}")
print(f"KS4 dir    : {ks_dir}")
print(f"Raw binary : {raw_file}")
print(f"Meta file  : {meta_file}")


#%%
# ── run Bombcell quality metrics ───────────────────────────────────────────────
import bombcell as bc

save_path = ks_dir / "bombcell"

param = bc.get_default_parameters(
    ks_dir,
    raw_file=raw_file,
    meta_file=meta_file,
    kilosort_version=4,
)
param["rpv_method"] = RPV_METHOD
param["extractRaw"] = True   # required to compute waveform-based metrics

quality_metrics, param, unit_type, unit_type_string = bc.run_bombcell(
    ks_dir, save_path, param
)

print(f"\nBombcell done. Results saved to: {save_path}")
import pandas as pd
ut = pd.Series(unit_type_string)
for label, count in ut.value_counts().items():
    print(f"  {label:25s}: {count}")


#%%
# ── compute ephys properties (ACG, ISI, firing rate) ─────────────────────────
ephys_param = bc.get_ephys_parameters(ks_dir)
ephys_properties, ephys_param = bc.run_all_ephys_properties(
    ks_dir, ephys_param, save_path=save_path
)
print(f"Ephys properties saved to: {save_path}")


#%%
# ── cell-type classification ───────────────────────────────────────────────────
cell_types = bc.classify_and_plot_brain_region(
    ephys_properties, ephys_param, BRAIN_REGION
)

print(f"\nCell-type classification ({BRAIN_REGION}):")
ct_series = pd.Series(cell_types)
for label, count in ct_series.value_counts().items():
    print(f"  {str(label):30s}: {count}")


#%%
# ── print unit classification summary from TSV ────────────────────────────────
# Bombcell writes cluster_bc_unitType.tsv to ks_dir for Phy compatibility.
tsv_files = list(ks_dir.glob("cluster_bc_unitType.tsv"))
if not tsv_files:
    print(f"No TSV output found in {ks_dir} — check that run_bombcell completed successfully.")
else:
    df = pd.read_csv(tsv_files[0], sep="\t")
    counts = df["bc_unitType"].value_counts()
    print(f"\nUnit classification (cluster_bc_unitType.tsv):")
    for label, count in counts.items():
        print(f"  {label:25s}: {count}")
    print(f"  {'TOTAL':25s}: {len(df)}")

# %%
