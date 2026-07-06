# Generate Kilosort channel map (.mat) files from an Open Ephys recording folder.
# Saves one .mat per ephys probe stream next to structure.oebin.
# Skips files that already exist.


#%%
# list channel indices to exclude from sorting (dead / noise / out-of-brain)
# 0-indexed, identical with the channel numbers in OpenEphys
bad_channels = [154]   # e.g. [72, 154, 203]


#%%
# path and params
from pathlib import Path
base_path = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14")


#%%
import sys
from IPython.core.getipython import get_ipython
sys.path.append(str(Path(__file__).parent.parent) if "__file__" in dir() else "..")

from src import oe_parse_folders, oe_parse_params, save_kilosort_channel_map

oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, _ = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])

print(f"Recording : {base_path.name}")
print(f"Probes found: {len(probes_params)}")
for p in probes_params:
    print(f"  {p['stream_name']}  —  {p['probe_name']}  ({p['channel_count']} ch)")


#%%
# generate channel map for each ephys probe
for probe_params in probes_params:
    save_kilosort_channel_map(probe_params, oe_paths["oebin"].parent, base_path.name,
                              bad_channels=bad_channels, overwrite=True)


# %%
