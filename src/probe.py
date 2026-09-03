"""
Neuropixels 2.0 Multi-shank Electrode Mapping
==============================================
Functions to parse the official Neuropixels 2.0 electrode-channel mapping
spreadsheet and build structured representations of the probe layout.

Data source: Neuropix_2_0_Electrode-Channel-mapping.xlsx
Sheets used: 'Multi-shank Shank 0' through 'Multi-shank Shank 3'

Terminology
-----------
channel  : recording channel index (0–383), fixed hardware output
electrode: physical site index on the probe (0–1279 per shank)
bank     : one of 4 groups of electrodes on a shank (bank 0–3)
section  : one of 8 fixed 48-channel windows within a bank
           (channels [s*48 : (s+1)*48] for section s)
"""

import openpyxl
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

N_SHANKS     = 4
N_CHANNELS   = 384
N_BANKS      = 4
N_SECTIONS   = 8
SECTION_SIZE = 48   # channels per section slot (fixed, even if not all map)

ROW_PITCH        = 15        # µm, vertical spacing between electrode rows
ELECTRODE_COL_XS = [0, 32]  # µm, shank-local x positions of the two electrode columns (even/odd parity)
ELEC_PER_SHANK   = 1280      # electrodes per shank; OE global index = shank * 1280 + local

SHEET_NAMES  = [f"Multi-shank Shank {s}" for s in range(N_SHANKS)]
MAPPING_PATH = Path(__file__).parent.parent / "assets" / "Neuropix_2_0_Electrode-Channel-mapping.xlsx"


# --------------------------------------------------------------------------- #
# Function 1: build_channel_mapping
# --------------------------------------------------------------------------- #

def build_npx2_multishank_channel_mapping(filepath: Path = MAPPING_PATH) -> list:
    """
    Parse the multi-shank sheets and return the raw channel→electrode mapping.

    Parameters
    ----------
    filepath : str
        Path to the Neuropix_2_0_Electrode-Channel-mapping.xlsx file.

    Returns
    -------
    channel_mapping : list[list[list[int | None]]]
        Shape: [N_SHANKS][N_CHANNELS][N_BANKS]

        channel_mapping[shank][channel][bank]
            = electrode index (int) if a physical site is connected, else None.

        Example:
            channel_mapping[0][0]  →  [0, 384, 768, 1152]
            channel_mapping[0][48] →  [288, 672, 1056, None]
    """
    wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)

    channel_mapping = []

    for shank_idx, sheet_name in enumerate(SHEET_NAMES):
        ws = wb[sheet_name]

        # Read all data rows (row 3 onward = channel 0 onward; skip 2 header rows)
        # Columns: A=channel number, B=bank0, C=bank1, D=bank2, E=bank3
        rows = list(ws.iter_rows(
            min_row=3, max_row=2 + N_CHANNELS,
            min_col=1, max_col=1 + N_BANKS,
            values_only=True
        ))

        # Build [channel][bank] for this shank
        shank_map = []
        for row in rows:
            # row[0] is the channel number (used only as a sanity check)
            electrodes_per_bank = [row[b + 1] for b in range(N_BANKS)]
            shank_map.append(electrodes_per_bank)

        # Validate we got exactly 384 channels
        assert len(shank_map) == N_CHANNELS, (
            f"Expected {N_CHANNELS} channels in {sheet_name}, got {len(shank_map)}"
        )

        channel_mapping.append(shank_map)

    wb.close()
    return channel_mapping


# --------------------------------------------------------------------------- #
# Function 2: build_channel_mapping_structure
# --------------------------------------------------------------------------- #

def build_npx2_multishank_channel_structure(channel_mapping: list) -> list:
    """
    Derive the probe structure (shank → bank → section → electrodes) from
    the raw channel mapping.

    Sections are defined by fixed 48-channel windows:
        section s  →  channels [s*48 : (s+1)*48]   (s = 0 … 7)

    Within each window, only channels that actually map to an electrode on
    that bank are included; the rest are dropped. A fully empty section (no
    electrode on this bank in this channel window) is represented as [].

    Parameters
    ----------
    channel_mapping : list[list[list[int | None]]]
        Output of build_channel_mapping().
        Shape: [N_SHANKS][N_CHANNELS][N_BANKS]

    Returns
    -------
    structure : list[list[list[list[int]]]]
        Shape: [N_SHANKS][N_BANKS][N_SECTIONS]

        structure[shank][bank][section]
            = sorted list of electrode indices present in that section slot.
              Empty list [] if no electrodes map to this section.

        Sections are ordered by ascending channel index (section 0 = ch 0–47,
        section 7 = ch 336–383), which already gives ascending channel order
        within each bank.

        Example (Shank 0, Bank 0):
            section 0 → [0, 1, …, 47]          (ch 0–47,   electrodes 0–47)
            section 1 → [288, 289, …, 335]      (ch 48–95,  electrodes 288–335)
            section 2 → [48, 49, …, 95]         (ch 96–143, electrodes 48–95)
            …
    """
    structure = []

    for shank_idx in range(N_SHANKS):
        shank_structure = []

        for bank_idx in range(N_BANKS):
            bank_sections = []

            for sec_idx in range(N_SECTIONS):
                ch_start = sec_idx * SECTION_SIZE
                ch_end   = ch_start + SECTION_SIZE  # exclusive

                # Collect all non-None electrodes in this channel window
                electrodes = [
                    channel_mapping[shank_idx][ch][bank_idx]
                    for ch in range(ch_start, ch_end)
                    if channel_mapping[shank_idx][ch][bank_idx] is not None
                ]

                bank_sections.append(electrodes)

            shank_structure.append(bank_sections)

        structure.append(shank_structure)

    return structure


# --------------------------------------------------------------------------- #
# Quick verification / demo
# --------------------------------------------------------------------------- #

def print_structure_summary(structure: list) -> None:
    """Print a compact summary of the probe structure."""
    for shank_idx, shank in enumerate(structure):
        print(f"\nShank {shank_idx}")
        for bank_idx, bank in enumerate(shank):
            parts = []
            for sec_idx, electrodes in enumerate(bank):
                n = len(electrodes)
                if n == 0:
                    parts.append("  --")
                else:
                    parts.append(f"{electrodes[0]:4d}-{electrodes[-1]:<4d}({n:2d})")
            print(f"  Bank {bank_idx}: " + " | ".join(parts))


def parse_imro_np2_multishank(path) -> dict:
    """
    Parse a NP2.0 multi-shank (probe type 2013/2014) .imro file.

    Each entry format: (channel shank bank refid local_electrode)
    Probe types 2013 and 2014 share this format (2014 adds a dovetail cap).

    Parameters
    ----------
    path : str or Path
        Path to the .imro file.

    Returns
    -------
    channel_electrode : dict {ch: global_electrode_index}
        global_electrode_index = shank * ELEC_PER_SHANK + local_electrode
    """
    text = Path(path).read_text().strip()
    entries = text.split(")(")[1:]
    channel_electrode = {}
    for entry in entries:
        fields = entry.rstrip(")").split()
        ch, shank, local_e = int(fields[0]), int(fields[1]), int(fields[4])
        channel_electrode[ch] = shank * ELEC_PER_SHANK + local_e
    return channel_electrode


if __name__ == "__main__":
    import sys

    filepath = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Neuropix_2_0_Electrode-Channel-mapping.xlsx"
    )

    print(f"Reading: {filepath}")
    mapping   = build_npx2_multishank_channel_mapping(filepath)
    structure = build_npx2_multishank_channel_structure(mapping)

    # --- spot-checks on channel_mapping ---
    print("\n=== channel_mapping spot-checks (Shank 0) ===")
    for ch in [0, 48, 96, 192]:
        print(f"  ch {ch:3d}: banks = {mapping[0][ch]}")

    # --- full structure summary ---
    print("\n=== channel_mapping_structure summary ===")
    print_structure_summary(structure)

    # --- verify section sizes ---
    print("\n=== Non-standard section sizes ===")
    for s in range(N_SHANKS):
        for b in range(N_BANKS):
            for sec in range(N_SECTIONS):
                n = len(structure[s][b][sec])
                if n not in (0, SECTION_SIZE):
                    print(f"  Shank {s}, Bank {b}, Section {sec}: {n} electrodes")


def find_contiguous_channel_groups(probe_params):
    """
    Group recorded channels by shank, then find runs of channels that are
    contiguous in the Y-axis.

    Contiguous = ypos gap ≤ 2 * ROW_PITCH (30 µm), which covers:
      - Same-row two-column selection (ypos diff 0)
      - Adjacent-row selection (ypos diff 15 µm)
      - Zig-zag single-column selection (ypos diff 30 µm, skips one row)
    A gap > 30 µm signals a true break in coverage.

    Parameters
    ----------
    probe_params  : dict  — must contain "channel_shank" and "channel_ypos"

    Returns
    -------
    groups : list of length N_SHANKS
        groups[shank] = list of channel-index lists, each list is one
        contiguous run sorted by ypos (ascending).
    """
    MAX_GAP = 2 * ROW_PITCH

    channel_shank = probe_params["channel_shank"]
    channel_ypos  = probe_params["channel_ypos"]

    shank_channels = [[] for _ in range(N_SHANKS)]
    for ch in channel_shank:
        shank_channels[channel_shank[ch]].append((channel_ypos[ch], ch))

    groups = []
    for shank_idx in range(N_SHANKS):
        pairs = sorted(shank_channels[shank_idx])
        if not pairs:
            groups.append([])
            continue

        runs = []
        current_run = [pairs[0][1]]
        for i in range(1, len(pairs)):
            if pairs[i][0] - pairs[i - 1][0] <= MAX_GAP:
                current_run.append(pairs[i][1])
            else:
                runs.append(current_run)
                current_run = [pairs[i][1]]
        runs.append(current_run)

        groups.append(runs)

    return groups