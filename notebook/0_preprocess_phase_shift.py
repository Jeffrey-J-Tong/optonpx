# Preprocess ephys binary with phase shift correction + CAR using SpikeInterface.
# Output is a binary file that can be fed directly to run_kilosort.py
# (set do_CAR=False and point dat_file to the output binary when running KS4).


#%%
# ── paths and params ───────────────────────────────────────────────────────────
from pathlib import Path

base_path  = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14")
PROBE_IDX  = 0       # which ephys stream to preprocess
STREAM_NAME = None   # e.g. "ProbeA"; None = auto-detect from OE folder

# output: preprocessed binary will be saved here
# recommended: a subfolder next to the recording so KS4 results stay organised
output_dir = base_path / "preprocessed"

# preprocessing options
DO_CAR     = True    # common average reference after phase shift (recommended)
CAR_MODE   = "global"   # "global" or "local"
CAR_OP     = "median"   # "median" or "average"


#%%
# ── resolve stream name from OE folder if not specified ───────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent) if "__file__" in dir() else "..")

from src import oe_parse_folders, oe_parse_params

oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, _   = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])

probe_params = probes_params[PROBE_IDX]
stream_name  = STREAM_NAME or probe_params["stream_name"]

print(f"Recording   : {base_path.name}")
print(f"Stream      : {stream_name}  ({probe_params['channel_count']} ch,  {probe_params['sample_rate']} Hz)")
print(f"Output dir  : {output_dir}")


#%%
# ── load recording with SpikeInterface ────────────────────────────────────────
import spikeinterface.full as si

# read_openephys loads probe geometry and inter-sample delays from OE metadata
recording = si.read_openephys(base_path, stream_name=stream_name)

print(f"Loaded: {recording.get_num_channels()} ch  x  {recording.get_num_samples()} samples")
print(f"Sample rate: {recording.get_sampling_frequency()} Hz")


#%%
# ── apply phase shift correction ──────────────────────────────────────────────
# corrects inter-channel time offsets from NP2.0 multiplexed ADC
recording_ps = si.phase_shift(recording)
print("Phase shift correction applied.")

if DO_CAR:
    recording_pre = si.common_reference(recording_ps,
                                        reference=CAR_MODE,
                                        operator=CAR_OP)
    print(f"CAR applied: {CAR_MODE} {CAR_OP}.")
else:
    recording_pre = recording_ps


#%%
# ── save preprocessed binary ──────────────────────────────────────────────────
# saves as int16 binary + a JSON sidecar with channel/probe metadata
output_dir.mkdir(parents=True, exist_ok=True)

recording_pre.save_to_folder(folder=output_dir, overwrite=True,
                             chunk_duration="1s", n_jobs=4,
                             progress_bar=True)

# the binary file KS4 needs
dat_file = output_dir / "traces_cached_seg0.raw"
n_chan_bin = recording_pre.get_num_channels()

print(f"\nDone.")
print(f"Binary : {dat_file}")
print(f"n_chan  : {n_chan_bin}")
print(f"\nIn run_kilosort.py, set:")
print(f"  dat_file  = Path(r'{dat_file}')")
print(f"  n_chan_bin = {n_chan_bin}   # already set from probe_params, should match")
print(f"  do_CAR    = False          # CAR already applied here")
