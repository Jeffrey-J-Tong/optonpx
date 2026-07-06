"""
Kilosort 4 channel map utilities for Neuropixels 2.0 data recorded with Open Ephys.
"""

from pathlib import Path

import numpy as np
import scipy.io


def save_kilosort_channel_map(
    probe_params: dict,
    output_dir: Path,
    recording_name: str,
    bad_channels: list[int] | None = None,
    overwrite: bool = False,
) -> Path | None:
    """
    Build a Kilosort 4 probe dictionary and save as .mat.

    Field names follow the KS4 convention (xc/yc, 0-indexed chanMap).
    Bad channels are excluded from all arrays — KS4 has no 'connected' field.

    X coordinates are taken as-is from settings.xml (absolute probe-level coords,
    shanks already offset from each other so KS4 displays them without overlap).

    Skips saving if the .mat file already exists.

    Parameters
    ----------
    probe_params   : dict         — one entry from probes_params (output of oe_parse_params)
    output_dir     : Path         — directory where the .mat file will be written
    recording_name : str          — filename prefix (e.g. base_path.name)
    bad_channels   : list[int]    — channel indices to exclude (default: none)

    Returns
    -------
    Path to the .mat file, or None if skipped.
    """
    probe_label = probe_params["stream_name"]   # e.g. "ProbeA"
    mat_path = Path(output_dir) / f"{recording_name}_{probe_label}.mat"

    if mat_path.exists():
        if not overwrite:
            print(f"Skipping {mat_path.name} (already exists)")
            return None
        print(f"Overwriting {mat_path.name}")

    bad = set(bad_channels) if bad_channels else set()
    channels = sorted(ch for ch in probe_params["channel_shank"] if ch not in bad)
    n = len(channels)

    xc      = np.array([probe_params["channel_xpos"][ch]  for ch in channels], dtype=float)
    yc      = np.array([probe_params["channel_ypos"][ch]  for ch in channels], dtype=float)
    kcoords = np.array([probe_params["channel_shank"][ch] for ch in channels], dtype=float)  # 0-indexed

    # chanMap must store actual 0-indexed row positions in the binary file (1-indexed in .mat;
    # load_probe subtracts 1). Using sequential 1..n would cause KS4 to read the wrong rows
    # when bad channels are excluded — each entry must be the original channel index + 1.
    chan_map = np.array([ch + 1 for ch in channels], dtype=float)

    # .mat format uses old KS field names — load_probe() converts to xc/yc/etc internally
    scipy.io.savemat(mat_path, {
        "chanMap":   chan_map.reshape(-1, 1),
        "xcoords":   xc.reshape(-1, 1),
        "ycoords":   yc.reshape(-1, 1),
        "kcoords":   kcoords.reshape(-1, 1),
        "connected": np.ones((n, 1), dtype=bool),
    })

    print(f"Saved {mat_path.name}  ({n} channels, {len(bad)} excluded)")
    return mat_path
