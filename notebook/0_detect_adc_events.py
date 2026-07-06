# This script is used to read all the ADC data and detect all the events in different channels.


#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.core.getipython import get_ipython
from pprint import pprint
sys.path.append("..")
ipython = get_ipython()
if ipython is not None:
    print(f"This script is running in an IPython environment.")


#%% basic recording info
from src import oe_parse_folders, oe_parse_params
base_path = Path(r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-17_13-28-04_manualsurvey")
oe_names, oe_paths = oe_parse_folders(base_path)
probes_params, adc_params = oe_parse_params(oe_paths["xml"], oe_paths["oebin"])


#%%
# load ADC data - this part should not be here ---------------------------------------------------
from src import oe_load_adc, oe_detect_adc_events
adc_data, adc_t, adc_fs = oe_load_adc(oe_paths["adc_stream"], adc_params)
print(f"ADC: {adc_data.shape[1]} channels, {adc_data.shape[0]} samples, {adc_fs/1000:.1f} kHz")
print(f"Duration: {adc_t[-1] - adc_t[0]:.2f} s  ({adc_t[0]:.3f} – {adc_t[-1]:.3f} s)")
print(f"Overall voltage range: {adc_data.min():.3f} – {adc_data.max():.3f} V")

# find events in certain channel
adc_channel_arduino = 2
events = oe_detect_adc_events(adc_data, adc_t, channel=adc_channel_arduino)
try:
    print(f"Duration range: {(events[:, 1] - events[:, 0]).min()*1000:.2f} – {(events[:, 1] - events[:, 0]).max()*1000:.2f} ms")
    print(events[:5])
except:
    print(f"No ADC event in channel {adc_channel_arduino}.")