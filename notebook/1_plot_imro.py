#%%
%whos

#%%
import sys
sys.path.append("..")
from pathlib import Path
from src import parse_imro_np2_multishank

imro_path = Path("testdata/0612.imro")
imro_mapping = parse_imro_np2_multishank(imro_path)
print(f"Parsed {len(imro_mapping)} channels")
print("Sample entries (ch: global_electrode):")
for ch in sorted(imro_mapping)[:8]:
    print(f"  ch {ch:3d} → electrode {imro_mapping[ch]}")

#%%
# option 1: show figure inline, using svg format to improve sharpness
%matplotlib inline
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'svg'
# option 2: plot in separate figure window, with tk or qt backend
# %matplotlib tk
# %matplotlib qt
from src import plot_electrode_map
fig = plot_electrode_map(imro_mapping)
# fig = plot_electrode_map()
# plt.show()
