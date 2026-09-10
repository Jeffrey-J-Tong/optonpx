# `metadata/`

Structured metadata for organising raw Open Ephys recordings and the
concatenated ("merged") recordings built from them. Loaded by
`src/metadata.py` (`load_session_metadata`, `load_merge_recipe`) and read by
`notebook/2.2_concatenate_sessions.ipynb`.

Raw recordings are **never modified** — every path in these files points at
read-only source data; all preprocessing lands in the merge `output_dir`.

## Two file types

### Session YAML — one per raw Open Ephys recording folder

`<animal>_<date>_<type>.yaml`

| field | required | notes |
|---|---|---|
| `session_id` | yes | unique string |
| `session_type` | yes | `behavior` or `optotagging` |
| `open_ephys.folder_path` | yes | the auto-named OE folder (has `Record Node .../experiment.../recording...`) |
| `open_ephys.experiment_number` | no | auto-detected if omitted |
| `probe_streams` | yes | list of ephys stream names, e.g. `[ProbeA]` or `[ProbeA, ProbeB]` |
| `adc.sync_1hz` / `adc.port_visits` | no | column index into the OneBox-ADC stream (= the `ADCn` number) |
| `animal` / `date` / `description` | no | free-form bookkeeping |
| `optotagging` | only for `optotagging` sessions | see below |

`optotagging` block:

- `laser_inputs` (**required**): list of `{color, adc_channel}`. `adc_channel`
  is the column index into the OneBox-ADC stream. Current rig: **ADC1 = blue,
  ADC3 = red** (ADC0 = 1 Hz sync, ADC2 = port visits).
- `pulse_detection`, `lag_estimation`, `artifact_mask`: optional; any omitted
  key falls back to the Berke-lab-pipeline default
  (`src/preprocess.py` `DEFAULT_PULSE_DETECTION` / `DEFAULT_LAG_ESTIMATION` /
  `DEFAULT_ARTIFACT_MASK`). These reproduce Dan's
  `load_laser_pulses_from_oe_adc.m` / `estimate_laser_artifact_lag_from_L.m` /
  `mask_probe_dat_artifacts.m`.

### Merge recipe YAML — one per concatenated output

`merge_<animal>_<date>.yaml`

| field | required | notes |
|---|---|---|
| `merge_id` | yes | unique string; names the output `.mat` files |
| `output_dir` | yes | where the concatenated `<stream>/continuous.dat` + sidecars go (must be outside every source folder) |
| `probe_streams` | yes | one output `continuous.dat` per stream; every stream must be in every session's `probe_streams` |
| `sessions` | yes | list of `{metadata: <session yaml, relative to this file>}`, **in concatenation order** — behavior first |
| `phase_shift.apply` | no (default `true`) | |
| `phase_shift.margin_ms` | no (default `40`) | FFT edge context per chunk |
| `phase_shift.chunk_s` | no (default `10`) | streamed chunk size; lower for less RAM |
| `kilosort_hint.do_car.<stream>` | no | written to `preprocessing.json` only; the notebook does not run CAR (Kilosort does, per shank) |

A recipe with a single `sessions` entry just phase-shifts (and masks, if
optotagging) that one session — no concatenation.

## Output of a merge

```
<output_dir>/
  <stream>/continuous.dat            # broadband int16, mask + phase shift applied
  <stream>/<merge_id>_<stream>.mat   # Kilosort channel map
  session_boundaries.json            # per stream: sample_start/end, fs, n_chan, source paths/sizes
  preprocessing.json                 # mask/lag/phase-shift params + provenance + kilosort_hint
  laser_events.csv                   # masked pulses: concat-axis sample indices + colour + session
```

`session_boundaries.json` is what lets a later step slice the combined spike
sorting back to one session (`sorting.frame_slice(sample_start, sample_end)`).
