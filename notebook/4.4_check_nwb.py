# %%
from pynwb import NWBHDF5IO
from pathlib import Path
path_nwb = Path(r"E:\D1-1-4_IM-1971\nwb\IM-1971_20260619_2_old_zarr\IM-1971_20260619.nwb")
with NWBHDF5IO(path_nwb, mode="r") as io:
    file_nwb = io.read()
    units = file_nwb.units
    print(len(units))
    print(units.colnames)
    df = file_nwb.units.to_dataframe()

# %%
df.head()
df["phy_group"].value_counts()
# %%
