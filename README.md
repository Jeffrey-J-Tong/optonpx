# OptoNPX

A Python toolkit for processing Neuropixels 2.0 multi-shank electrophysiology
recordings made with Open Ephys (OneBox hardware), from raw binary through
spike sorting, curation, and brain-region assignment.

> **Draft README** — generated from the current codebase; please review and
> correct anything that doesn't match your actual workflow before relying on it.

## Pipeline overview

The `notebook/` notebooks are numbered in the order they're meant to be run.
Each is a Jupyter `.ipynb` — open it in VS Code / Jupyter and run cell by
cell.

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `0_parse_channel_mapping.py` | Parse the NP2.0 electrode-channel mapping spreadsheet |
| 0 | `0_detect_adc_events.py` | Detect events (e.g. laser pulses) on ADC channels |
| 0 | `0_bad_channel_detection.py` | Flag bad channels before sorting |
| 0 | `0_preprocess_phase_shift.py` | Phase-shift correction + CAR preprocessing (SpikeInterface), writes a binary for Kilosort |
| 1 | `1_plot_imro.py` | Plot the probe's `.imro` channel map |
| 1 | `1_visualize_raw_trace.py` | Visualize the channel map and raw/processed traces |
| 2.1 | `2.1_generate_channel_map.py` | Generate Kilosort 4 channel-map `.mat` files from an OE recording |
| 2.2 | `2.2_run_kilosort.py` | Run Kilosort 4 spike sorting |
| 3.1 | `3.1_run_bombcell.py` | Run Bombcell quality metrics / cell-type classification |
| 3.2 | `3.2_run_unitrefine.py` | Run UnitRefine automatic unit curation |
| 4.1 | `4.1_curation_summary.py` | Compare KSLabel / Bombcell / manual Phy curation agreement |
| 4.2 | `4.2_verify_analyzer_zarr_status.py` | Inspect a SortingAnalyzer zarr for consistency with Phy curation |
| 4.3 | `4.3_analyzer_load_curation.py` | Build a SortingAnalyzer from Phy-curated KS4 output |
| 4.4 | `4.4_check_nwb.py` | Inspect an exported NWB file |
| 5.1 | `5.1_assign_brain_regions.py` | Interactively assign a brain region to each channel on the electrode map |

Typical flow: **parse & preprocess raw data → run Kilosort 4 → run automated
QC (Bombcell / UnitRefine) → manual curation in Phy → build an analyzer →
assign brain regions**.

## Environments

Different pipeline steps depend on separate conda environments (some tools
have conflicting dependencies):

| Environment | Used by | Notes |
|-------------|---------|-------|
| `optonpx` | Steps 0, 1, 2.1, 4.1, 5.1 | Core toolkit — `conda activate optonpx` |
| Kilosort env | Step 2.2 | See the Kilosort 4 install docs |
| `bombcell` | Step 3.1 | `conda activate bombcell` |
| `unitrefine`/SpikeInterface env | Step 3.2 | See UnitRefine install docs |
| `spikeinterface` | Step 4.2, 4.3 | `conda activate spikeinterface` |

`optonpx` dependencies (`requirements.txt`): `numpy`, `scipy`, `pandas`,
`matplotlib`, `ipywidgets`, `jupyterlab`. Also required but not yet listed
there: `openpyxl` (probe mapping spreadsheet) and `panel` + `bokeh`
(brain-region assignment tool, step 5.1) — install with:

```bash
conda activate optonpx
pip install -r requirements.txt
pip install openpyxl panel bokeh
```

## Repository layout

```
src/                    Core library code (see below)
notebook/               .ipynb pipeline notebooks, numbered by stage
assets/                 Static data (e.g. NP2.0 electrode-channel mapping spreadsheet)
references/             Reference material
old/                    Deprecated code — not used, do not reference
```

### `src/`

- **`openephys.py`** — parses an Open Ephys recording folder (one Record
  Node / experiment / recording) into paths and probe/ADC parameters; loads
  continuous binary data and detects ADC events.
- **`probe.py`** — parses the NP2.0 electrode-channel mapping spreadsheet
  into channel→electrode mappings and per-shank/bank/section structure.
- **`viz.py`** — electrode map plotting (`plot_electrode_map`).
- **`kilosort_helper.py`** — builds and saves Kilosort 4 channel-map `.mat`
  files.
- **`region_map.py`** — interactive Panel + Bokeh app for assigning a brain
  region to each recorded channel, plus JSON save/load for the result.

`src/__init__.py` re-exports the public API used across the `notebook/`
scripts.

## Notebooks

Notebooks in `notebook/` are Jupyter `.ipynb` files, run cell by cell in
VS Code / Jupyter. Saved cell outputs are kept so results and tracebacks
stay inspectable later.

## NP2.0 hardware reference

- 4 shanks, 384 recorded channels, 4 banks per shank, 8 sections per bank,
  48 channels per section.
- 1280 electrode sites per shank (banks 0–3 cover 320 sites each; bank 3
  has only 128 usable sites on shanks 0/1, with 32 spare slots).

See `CLAUDE.md` for more detailed conventions and architecture notes used
when developing this codebase with Claude Code.
