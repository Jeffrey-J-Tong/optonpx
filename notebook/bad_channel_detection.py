#%%
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append("..")

import numpy as np
from pathlib import Path
from collections import Counter

#%%
from src import oe_parse_folders, oe_parse_params, oe_load_adc

# --- recording params ---
base_path = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-18_17-19-35_manualsurvey_shank1_bankAB")
oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, adc_params = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])
probe = probes_params[0]
print(f"Probe: {probe['probe_name']}  s/n {probe['serial_number']}")
print(f"Channels: {probe['channel_count']}  sample rate: {probe['sample_rate']} Hz")

# --- t0_rec ---
adc_data, adc_t, adc_fs = oe_load_adc(oe_paths["adc_stream"], adc_params)
t0_rec = adc_t[0]
print(f"t0_rec: {t0_rec:.3f} s")

#%%
probe_dir = oe_paths["ephys_streams"][0]
recording_dir = oe_paths["oebin"].parent  # folder containing structure.oebin

# --- SI bad channel detection ---
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

# SI stream names have format "Record Node XXX#OneBox-YYY.ProbeA"
# probe["stream_name"] is just the suffix (e.g. "ProbeA") — find the full name
si_stream_names = se.OpenEphysBinaryRecordingExtractor.get_streams(recording_dir)[0]
si_stream_name = next(s for s in si_stream_names if s.endswith(probe["stream_name"]))
print(f"SI stream: {si_stream_name}")

recording = se.read_openephys(recording_dir, stream_name=si_stream_name)
print(recording)

bad_channel_ids, si_labels = spre.detect_bad_channels(
    recording,
    method="coherence+psd",
)

print(f"Bad channels detected: {len(bad_channel_ids)}")
print(Counter(si_labels))

#%%
# --- build channel_labels dict ---
channel_labels = {}
for ch_idx, label in enumerate(si_labels):
    if label != "good":
        channel_labels[ch_idx] = label

out_chs   = sorted(ch for ch, lbl in channel_labels.items() if lbl == "out")
dead_chs  = sorted(ch for ch, lbl in channel_labels.items() if lbl == "dead")
noise_chs = sorted(ch for ch, lbl in channel_labels.items() if lbl == "noise")
print(f"out  ({len(out_chs)}):   {out_chs}")
print(f"dead ({len(dead_chs)}):  {dead_chs}")
print(f"noise({len(noise_chs)}): {noise_chs}")

# %%
