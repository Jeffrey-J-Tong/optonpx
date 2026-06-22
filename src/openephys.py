from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
import numpy as np


@dataclass
class EphysRawSlice:
    data        : np.ndarray   # (n_channels, n_samples_with_pad), µV, float32
    t_arr       : np.ndarray   # (n_samples_with_pad,), seconds, OE clock
    valid_slice : slice        # trim after filtering: data[:, valid_slice]
    sample_rate : float


def oe_load_ephys_slice(ephys_dir, probe_params, t_start, t_end, pad_s=0.0,
                        is_relative_time=False):
    """
    Load a time window of ephys continuous data.

    Parameters
    ----------
    ephys_dir    : Path  — oe_paths["ephys_streams"][i]
    probe_params : dict  — probes_params[i] from oe_parse_params
    t_start, t_end : float — requested window in seconds.
        If is_relative_time=False (default): OE clock timestamps.
        If is_relative_time=True: seconds from the first sample of this stream.
    pad_s        : float — extra data on each side for filtering (seconds)
    is_relative_time : bool — if True, t_start/t_end are relative to recording
        start and t0_rec is added internally (no need to read ADC timestamps)

    Returns
    -------
    EphysRawSlice
        .data        (n_ch, n_samples_with_pad), µV, float32
        .t_arr       (n_samples_with_pad,), seconds (OE clock)
        .valid_slice slice to apply after filtering
        .sample_rate float
    """
    ephys_dir = Path(ephys_dir)
    n_ch = int(probe_params["channel_count"])
    bv   = float(probe_params["channel_bit_volts"])
    fs   = float(probe_params["sample_rate"])

    t_all  = np.load(ephys_dir / "timestamps.npy", mmap_mode="r")
    t0_rec = float(t_all[0])

    if is_relative_time:
        t_start = t_start + t0_rec
        t_end   = t_end   + t0_rec

    pad_n   = int(round(pad_s * fs))
    i_start = max(0, int(round((t_start - t0_rec) * fs)) - pad_n)
    i_end   = min(len(t_all), int(round((t_end - t0_rec) * fs)) + pad_n)

    raw = np.memmap(ephys_dir / "continuous.dat", dtype="int16", mode="r")
    raw = raw.reshape(len(t_all), n_ch)[i_start:i_end, :]

    data  = (raw.astype("float32") * bv).T
    t_arr = np.array(t_all[i_start:i_end])

    pad_left  = int(round((t_start - t0_rec) * fs)) - i_start
    pad_right = i_end - int(round((t_end - t0_rec) * fs))
    valid_slice = slice(pad_left, data.shape[1] - pad_right if pad_right > 0 else None)

    return EphysRawSlice(data=data, t_arr=t_arr, valid_slice=valid_slice, sample_rate=fs)


def oe_load_adc(adc_dir, adc_params):
    """
    Load continuous binary data from an Open Ephys stream directory.

    Parameters
    ----------
    adc_dir : Path
        oe_paths["adc_stream"] — directory containing continuous.dat and timestamps.npy.
    adc_params : dict
        adc_params from parse_onebox_params — must have "channel_count", "channel_bit_volts", "sample_rate".

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_channels), float32, volts
    t_arr : np.ndarray, shape (n_samples,), float64, seconds (OE clock, does not start at 0)
    sample_rate : float
    """
    adc_dir = Path(adc_dir)
    n_ch = int(adc_params["channel_count"])
    bv   = float(adc_params["channel_bit_volts"])

    raw = np.memmap(adc_dir / "continuous.dat", dtype="int16", mode="r")
    n_samples = raw.size // n_ch
    raw = raw.reshape(n_samples, n_ch)

    data        = (raw * bv).astype("float32")
    t_arr       = np.load(adc_dir / "timestamps.npy")
    sample_rate = float(adc_params["sample_rate"])

    return data, t_arr, sample_rate


def oe_detect_adc_events(adc_data, adc_t, channel, threshold=4.0):
    """
    Detect threshold-crossing events in an ADC channel.

    Parameters
    ----------
    adc_data  : np.ndarray, shape (n_samples, n_channels)
    adc_t     : np.ndarray, shape (n_samples,), seconds (OE clock)
    channel   : int
    threshold : float, volts (default 4.0)

    Returns
    -------
    events : np.ndarray, shape (n_events, 2)
        Each row is [t_start, t_end] in seconds (OE clock).
    """
    sig   = adc_data[:, channel]
    above = sig >= threshold
    onsets  = np.where(~above[:-1] &  above[1:])[0] + 1
    offsets = np.where( above[:-1] & ~above[1:])[0] + 1

    if above[0]:
        onsets = np.concatenate([[0], onsets])
    if above[-1]:
        offsets = np.concatenate([offsets, [len(sig) - 1]])

    return np.column_stack([adc_t[onsets], adc_t[offsets]])


def oe_parse_folders(base_path: Path):

    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"base_path does not exist: {base_path}")
    oe_base = base_path.name

    # --- recording node ---
    node_dirs = [
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith("Record Node")
    ]
    if len(node_dirs) == 0:
        raise ValueError(f"No 'Record Node' directory found under: {base_path}")
    if len(node_dirs) > 1:
        raise ValueError(
            f"Multiple 'Record Node' directories found under: {base_path}\n"
            f"  Found: {[d.name for d in node_dirs]}\n"
            f"  Please check the folder structure."
        )
    node_dir = node_dirs[0]
    oe_node = node_dir.name

    # --- settings file ---
    xml_file = node_dir / "settings.xml"
    oe_xml = xml_file.name if xml_file.exists() else None

    # --- experiment ---
    exp_dirs = [
        d for d in node_dir.iterdir()
        if d.is_dir() and d.name.startswith("experiment")
    ]
    if len(exp_dirs) == 0:
        raise ValueError(f"No 'experiment' directory found under: {node_dir}")
    if len(exp_dirs) > 1:
        raise ValueError(
            f"Multiple 'experiment' directories found under: {node_dir}\n"
            f"  Found: {[d.name for d in exp_dirs]}\n"
            f"  Please check the folder structure."
        )
    exp_dir = exp_dirs[0]
    oe_experiment = exp_dir.name

    # --- recording ---
    rec_dirs = [
        d for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("recording")
    ]
    if len(rec_dirs) == 0:
        raise ValueError(f"No 'recording' directory found under: {exp_dir}")
    if len(rec_dirs) > 1:
        raise ValueError(
            f"Multiple 'recording' directories found under: {exp_dir}\n"
            f"  Found: {[d.name for d in rec_dirs]}\n"
            f"  Please check the folder structure."
        )
    rec_dir = rec_dirs[0]
    oe_recording = rec_dir.name

    # --- oebin file ---
    oebin_file = rec_dir / "structure.oebin"
    oe_oebin = oebin_file.name if oebin_file.exists() else None

    # --- continuous streams ---
    continuous_dir = rec_dir / "continuous"
    oe_ephys_streams, oe_adc_stream = [], None

    if continuous_dir.exists():
        for stream_dir in sorted(continuous_dir.iterdir()):
            if not stream_dir.is_dir():
                continue
            if "ADC" in stream_dir.name:
                oe_adc_stream = stream_dir.name
            else:
                oe_ephys_streams.append(stream_dir.name)

    oe_names = {
        "base":          oe_base,
        "node":          oe_node,
        "xml":           oe_xml,
        "experiment":    oe_experiment,
        "recording":     oe_recording,
        "oebin":         oe_oebin,
        "ephys_streams": oe_ephys_streams,
        "adc_stream":    oe_adc_stream,
    }

    oe_paths = {
        "base":           base_path,
        "node":           node_dir,
        "xml":            xml_file if xml_file.exists() else None,
        "experiment":     exp_dir,
        "recording":      rec_dir,
        "oebin":          oebin_file if oebin_file.exists() else None,
        "ephys_streams":  [continuous_dir / s for s in oe_ephys_streams],
        "adc_stream":     continuous_dir / oe_adc_stream if oe_adc_stream else None,
    }
    return oe_names, oe_paths

def oe_parse_xml_params(xml_path: Path):

    # read parameters from .xml file, extract probe and adc parameters from the OneBox processor node
    # probes parameters: list of dicts, one per probe
    # adc parameters: dict

    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    probes_params = []
    adc_params = {}

    node_signalchain = root.find("SIGNALCHAIN")
    node_processor = node_signalchain.find("PROCESSOR[@name='OneBox']")
    for node_stream in node_processor.findall("STREAM"):
        # pprint(f"Node tag: {node_stream.tag}")
        # pprint(f"Node attributes: {node_stream.attrib}")
        if node_stream.attrib.get("name") == "OneBox-ADC":
            adc_params = {
                "stream_name": node_stream.attrib.get("name"),
                "sample_rate": node_stream.attrib.get("sample_rate"),
                "channel_count": node_stream.attrib.get("channel_count"),
            }
        else:
            probes_params.append({
                "stream_name": node_stream.attrib.get("name"),
                "sample_rate": node_stream.attrib.get("sample_rate"),
                "channel_count": node_stream.attrib.get("channel_count"),
            })

    for i, node_stream in enumerate(root.findall(".//NP_PROBE")):
        # pprint(f"Node tag: {node_stream.tag}")
        # pprint(f"Node attributes: {node_stream.attrib}")
        node_channels = node_stream.find("CHANNELS")  # channels are "bank:shank"
        # pprint(f"CHANNELS tag: {node_channels.tag}")
        channel_bank = {}
        channel_shank = {}
        for key, val in node_channels.attrib.items():
            channel_num = int(key.replace("CH", ""))
            bank, shank = val.split(":")
            channel_bank[channel_num] = int(bank)
            channel_shank[channel_num] = int(shank)
        node_x_pos = node_stream.find("ELECTRODE_XPOS")
        channel_xpos = {}
        for key, val in node_x_pos.attrib.items():
            channel_num = int(key.replace("CH", ""))
            channel_xpos[channel_num] = int(val)
        node_y_pos = node_stream.find("ELECTRODE_YPOS")
        channel_ypos = {}
        for key, val in node_y_pos.attrib.items():
            channel_num = int(key.replace("CH", ""))
            channel_ypos[channel_num] = int(val)
        node_channel_electrode = node_stream.find("ELECTRODE_INDEX")
        channel_electrode = {}
        for key, val in node_channel_electrode.attrib.items():
            channel_num = int(key.replace("CH", ""))
            channel_electrode[channel_num] = int(val)
        probes_params[i].update({
            "serial_number": node_stream.attrib.get("probe_serial_number"),
            "part_number": node_stream.attrib.get("probe_part_number"),
            "port": node_stream.attrib.get("port"),
            "dock": node_stream.attrib.get("dock"),
            "probe_name": node_stream.attrib.get("probe_name"),
            "electrode_config_preset": node_stream.attrib.get("electrodeConfigurationPreset"),
            "reference": node_stream.attrib.get("referenceChannel"),
            "channel_bank": channel_bank,
            "channel_shank": channel_shank,
            "channel_xpos": channel_xpos,
            "channel_ypos": channel_ypos,
            "channel_electrode": channel_electrode,
        })

    return probes_params, adc_params

def oe_parse_oebin_params(oebin_path: Path):
    import json
    # print(f"Parsing oebin file: {oebin_path}")
    with open(oebin_path, "r") as f:
        oebin_data = json.load(f)
    probes_params = []
    adc_params = {}
    for oebin_continuous in oebin_data["continuous"]:
        if "Probe" in oebin_continuous["folder_name"]:
            channel_ypos = {}
            channel_electrode = {}
            channel_bit_volts = {}
            for oebin_channel in oebin_continuous["channels"]:
                channel_int = int(oebin_channel["channel_name"].replace("CH", ""))
                channel_ypos[channel_int] = int(oebin_channel["channel_metadata"][0]["value"][0])
                channel_electrode[channel_int] = int(oebin_channel["channel_metadata"][1]["value"][0])
                channel_bit_volts[channel_int] = float(oebin_channel["bit_volts"])
            if len(set(channel_bit_volts.values())) == 1:
                channel_bit_volts = channel_bit_volts[0]
            else:
                raise ValueError(f"Error: Probe channels have different bit_volts in oebin: {oebin_continuous['stream_name']}")
            probes_params.append({
                "stream_name": oebin_continuous["stream_name"],
                "sample_rate": oebin_continuous["sample_rate"],
                "channel_count": oebin_continuous["num_channels"],
                "channel_ypos": channel_ypos,
                "channel_electrode": channel_electrode,
                "channel_bit_volts": channel_bit_volts,
            })
        elif "ADC" in oebin_continuous["folder_name"]:
            channel_bit_volts = []
            for oebin_channel in oebin_continuous["channels"]:
                channel_int = int(oebin_channel["channel_name"].replace("ADC", ""))
                channel_bit_volts.append(float(oebin_channel["bit_volts"]))
            if len(set(channel_bit_volts)) == 1:
                channel_bit_volts = channel_bit_volts[0]
            else:
                raise ValueError(f"Error: ADC channels have different bit_volts in oebin: {oebin_continuous['stream_name']}")
            adc_params = {
                "stream_name": oebin_continuous["stream_name"],
                "channel_count": oebin_continuous["num_channels"],
                "sample_rate": oebin_continuous["sample_rate"],
                "channel_bit_volts": channel_bit_volts,
            }
        else:
            print(f"Unknown stream type in oebin: {oebin_continuous['folder_name']}")
    return probes_params, adc_params

def _values_equal(a, b):
    """Compare with numeric type coercion — XML stores numbers as strings."""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False

def merge_two_dicts(dict_a, dict_b, label="params"):
    """Merge two dicts; raise ValueError listing every conflicting key."""
    merged = dict(dict_a)
    conflicts = []
    for key, val_b in dict_b.items():
        if key in merged:
            if not _values_equal(merged[key], val_b):
                conflicts.append((key, merged[key], val_b))
        else:
            merged[key] = val_b
    if conflicts:
        lines = "\n".join(f"  {k!r}: xml={a!r}, oebin={b!r}" for k, a, b in conflicts)
        raise ValueError(f"Conflicting values in {label}:\n{lines}")
    return merged

def oe_parse_params(xml_path: Path, oebin_path: Path):
    """
    Parse and merge parameters from both .xml file and oebin file
    """
    xml_probes_params, xml_adc_params = oe_parse_xml_params(xml_path)
    oebin_probes_params, oebin_adc_params = oe_parse_oebin_params(oebin_path)
    # --- probe params (list of dicts, matched by stream_name) ---
    xml_by_name   = {p["stream_name"]: p for p in xml_probes_params}
    oebin_by_name = {p["stream_name"]: p for p in oebin_probes_params}

    if set(xml_by_name) != set(oebin_by_name):
        raise ValueError(
            f"Probe stream names don't match:\n"
            f"  xml:   {sorted(xml_by_name)}\n"
            f"  oebin: {sorted(oebin_by_name)}"
        )

    probes_params = [
        merge_two_dicts(xml_by_name[name], oebin_by_name[name], label=f"probe '{name}'")
        for name in xml_by_name
    ]

    # --- ADC params (single dict each) ---
    adc_params = merge_two_dicts(xml_adc_params, oebin_adc_params, label="ADC params")

    return probes_params, adc_params

