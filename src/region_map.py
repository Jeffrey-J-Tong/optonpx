"""
Interactive brain-region assignment for NP2.0 multi-shank channel maps.

Lets the user box-select / click-select recorded channels on the electrode
map, assign a brain-region label from a fixed dropdown, iterate, and save
the result to a per-channel JSON file. Built on Panel + Bokeh (same stack
PixelMap uses for its Neuropixels channelmap GUI) so selection, zoom, and
pan use Bokeh's built-in toolbar tools instead of hand-rolled matplotlib
event handlers, and the control panel uses real web widgets. Notebook-only
(no plain-script / non-notebook use case is required for this tool).
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
from bokeh.events import Tap
from bokeh.layouts import gridplot
from bokeh.models import (
    BoxSelectTool, ColumnDataSource, LabelSet, PanTool, Range1d,
    ResetTool, WheelZoomTool,
)
from bokeh.plotting import figure

from src.probe import (
    N_SHANKS, N_BANKS, N_SECTIONS, ROW_PITCH, ELECTRODE_COL_XS, ELEC_PER_SHANK,
    build_npx2_multishank_channel_mapping, build_npx2_multishank_channel_structure,
)

_mapping   = build_npx2_multishank_channel_mapping()
_structure = build_npx2_multishank_channel_structure(_mapping)

# --------------------------------------------------------------------------- #
# Placeholder region list — EDIT THIS to match your actual target regions.
# "unknown" must stay in the list; it is the default label for every channel.
# --------------------------------------------------------------------------- #
BRAIN_REGIONS = ["putative_NAc_core", "putative_dCA1", "putative_dCA3", "unknown"]

INACTIVE_COLOR    = "#cccccc"   # electrode site not recorded on this channel
UNASSIGNED_COLOR  = "black"     # recorded channel, still labeled "unknown"
SELECT_EDGE_COLOR = "#ff3b30"
SELECT_EDGE_WIDTH = 2.2
TRANSPARENT       = "rgba(0,0,0,0)"

RECT_W, RECT_H = 10, 10
BOX_X_PAD      = 16
LABEL_MARGIN   = 22
LABEL_INSET    = 3 * ROW_PITCH
CLICK_RADIUS   = 8.0   # µm; a Tap event toggles the nearest channel within this radius


def _region_colors(regions):
    """Categorical color per region ('unknown' always maps to UNASSIGNED_COLOR)."""
    cmap = plt.get_cmap("tab10")
    colors = {}
    i = 0
    for r in regions:
        if r == "unknown":
            continue
        colors[r] = mcolors.to_hex(cmap(i % 10))
        i += 1
    colors["unknown"] = UNASSIGNED_COLOR
    return colors


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #

def save_region_assignment(assignment: dict, path) -> None:
    """
    Save {channel: region} to a per-channel JSON file (IBL channel_locations.json
    inspired, but the field is named 'brain_region' rather than 'acronym' since
    no CCF atlas registration is involved — this is a user-defined label).
    """
    path = Path(path)
    out = {str(ch): {"brain_region": region} for ch, region in sorted(assignment.items())}
    path.write_text(json.dumps(out, indent=2))


def load_region_assignment(path) -> dict:
    """Load a {channel: region} dict from a previously saved JSON file."""
    path = Path(path)
    raw = json.loads(path.read_text())
    return {int(ch): entry["brain_region"] for ch, entry in raw.items()}


# --------------------------------------------------------------------------- #
# Geometry data-prep (plain data, no plotting library involved)
# --------------------------------------------------------------------------- #

def _build_layout(channel_electrode):
    """
    Re-derive the same per-shank electrode geometry used by plot_electrode_map,
    as plain dict/list data (no matplotlib patches / bokeh models here) so it
    can feed Bokeh ColumnDataSources.
    """
    mapping, structure = _mapping, _structure
    channels = sorted(channel_electrode.keys())

    ch_shank, ch_bank, ch_local_e = {}, {}, {}
    for ch in channels:
        sh = channel_electrode[ch] // ELEC_PER_SHANK
        local_e = channel_electrode[ch] % ELEC_PER_SHANK
        ch_shank[ch] = sh
        ch_local_e[ch] = local_e
        for bk in range(N_BANKS):
            if mapping[sh][ch][bk] == local_e:
                ch_bank[ch] = bk
                break

    active_banks  = sorted(set(ch_bank[ch] for ch in channels))
    banks_to_draw = list(range(max(active_banks) + 1))
    active_elec   = {(ch_shank[ch], ch_bank[ch], ch_local_e[ch]) for ch in channels}
    elec_to_ch    = {(ch_shank[ch], ch_bank[ch], ch_local_e[ch]): ch for ch in channels}

    row_pitch = ROW_PITCH
    half_step = row_pitch / 2
    sh_xs     = ELECTRODE_COL_XS

    drawn_ys = [
        (e // 2) * row_pitch
        for bk in banks_to_draw
        for sh in range(N_SHANKS)
        for sec in range(N_SECTIONS)
        for e in structure[sh][bk][sec]
    ]
    y_pad = 25
    y_lim = (min(drawn_ys) - half_step - y_pad, max(drawn_ys) + half_step + y_pad)
    x_lim = (min(sh_xs) - BOX_X_PAD - LABEL_MARGIN, max(sh_xs) + BOX_X_PAD + LABEL_MARGIN)

    shanks = []
    for shank_id in range(N_SHANKS):
        active, inactive = [], []
        boxes, dividers  = [], []
        labels_left, labels_right = [], []

        box_x0 = min(sh_xs) - BOX_X_PAD
        box_x1 = max(sh_xs) + BOX_X_PAD

        for bk in banks_to_draw:
            sec_y_ranges = {}
            for sec in range(N_SECTIONS):
                elecs = structure[shank_id][bk][sec]
                if not elecs:
                    continue
                ys_sec = [(e // 2) * row_pitch for e in elecs]
                sec_y_ranges[sec] = (min(ys_sec), max(ys_sec))
            if not sec_y_ranges:
                continue

            box_y0 = min(r[0] for r in sec_y_ranges.values()) - half_step
            box_y1 = max(r[1] for r in sec_y_ranges.values()) + half_step
            boxes.append((box_x0, box_y0, box_x1, box_y1))

            sorted_secs = sorted(sec_y_ranges.items(), key=lambda t: t[1][0])
            for i in range(len(sorted_secs) - 1):
                y_div = (sorted_secs[i][1][1] + sorted_secs[i + 1][1][0]) / 2
                dividers.append((box_x0, box_x1, y_div, y_div))

            for sec in range(N_SECTIONS):
                for e in structure[shank_id][bk][sec]:
                    ex = sh_xs[e % 2]
                    ey = (e // 2) * row_pitch
                    key = (shank_id, bk, e)
                    if key in active_elec:
                        ch = elec_to_ch[key]
                        active.append({"ch": ch, "x": ex, "y": ey})
                    else:
                        inactive.append({"x": ex, "y": ey})

            for sec in range(N_SECTIONS):
                for parity in range(2):
                    items = []
                    for ch in range(sec * 48, (sec + 1) * 48):
                        e = mapping[shank_id][ch][bk]
                        if e is None or e % 2 != parity:
                            continue
                        items.append(((e // 2) * row_pitch, ch))
                    if not items:
                        continue
                    items.sort()
                    y_label_bot = items[0][0] + LABEL_INSET
                    y_label_top = items[-1][0] - LABEL_INSET
                    if parity == 0:
                        lx = box_x0 - 2
                        labels_left.append((lx, y_label_bot, str(items[0][1])))
                        labels_left.append((lx, y_label_top, str(items[-1][1])))
                    else:
                        lx = box_x1 + 2
                        labels_right.append((lx, y_label_bot, str(items[0][1])))
                        labels_right.append((lx, y_label_top, str(items[-1][1])))

        shanks.append({
            "active": active, "inactive": inactive,
            "boxes": boxes, "dividers": dividers,
            "labels_left": labels_left, "labels_right": labels_right,
            "x_range": (box_x0 - LABEL_MARGIN, box_x1 + LABEL_MARGIN),
        })

    return {"channels": channels, "y_lim": y_lim, "shanks": shanks}


# --------------------------------------------------------------------------- #
# Interactive tool
# --------------------------------------------------------------------------- #

def build_region_assignment_app(channel_electrode, existing_path=None, save_path=None, title=None):
    """
    Interactive electrode map (Panel app) for assigning a brain region to
    each recorded channel.

    Parameters
    ----------
    channel_electrode : dict {ch: global_electrode_index}
        probe["channel_electrode"] from probes_params — the recorded channels.
    existing_path : Path or None
        If given and the file exists, preload the saved assignment (resume
        editing a previous session instead of starting from all-"unknown").
    save_path : Path or None
        Where the "Save" button writes the result. Defaults to existing_path.
    title : str or None

    Returns
    -------
    app : panel.Row
        Display it as the last expression of a notebook cell, or call
        `app.show()` to open it in a browser tab.
    get_assignment : callable() -> dict {ch: region}
        Retrieve the current in-memory assignment at any time.

    Controls
    --------
    - Drag a box (default toolbar mode) on any shank to ADD the enclosed
      channels to the selection. Click a single channel to TOGGLE it in/out
      (a plain click always works, independent of the toolbar's active tool
      — it's a raw Tap event handler with nearest-channel hit-testing, not
      a selectable Tool, since TapTool's "active" state proved unreliable
      once gridplot merges per-figure toolbars into one).
    - Scroll to zoom; switch the toolbar to the Pan tool to drag-pan (all
      four shanks stay in sync — they share a y-axis).
    - Pick a region in the dropdown (does not list "unknown"), click
      "Assign" to label the current selection.
    - "Set to unknown" resets the current selection back to "unknown"
      instead — the dropdown never offers "unknown" directly.
    - "Clear selection" drops the pending selection without labeling.
    - "Reset all to unknown" sets every channel back to "unknown", regardless
      of selection. Needs a second click to confirm — the first click warns
      and relabels itself "Confirm reset ALL".
    - A live "Assigned so far" panel shows the running count per region
      (color-swatched to match the map): "unknown" always listed first,
      other regions only once they have at least one channel.
    - "Save": if no channel is still "unknown", saves immediately.
      Otherwise the first click warns how many are still "unknown" and
      relabels itself "Confirm & Save" — click it again to save anyway.
    """
    pn.extension()

    save_path = Path(save_path) if save_path is not None else (
        Path(existing_path) if existing_path is not None else None
    )
    if save_path is None:
        raise ValueError("Provide save_path (or existing_path) so the Save button knows where to write.")

    layout = _build_layout(channel_electrode)
    channels = layout["channels"]

    # initial assignment ------------------------------------------------------
    if existing_path is not None and Path(existing_path).exists():
        assignment = load_region_assignment(existing_path)
        for ch in channels:
            assignment.setdefault(ch, "unknown")
    else:
        assignment = {ch: "unknown" for ch in channels}

    region_colors = _region_colors(BRAIN_REGIONS)
    selection = set()   # currently selected channels (pending assignment)

    active_sources = []   # one ColumnDataSource per shank, for cross-shank updates
    figs = []

    y_range = Range1d(*layout["y_lim"])

    for shank_id, sh in enumerate(layout["shanks"]):
        active_chs = [row["ch"] for row in sh["active"]]
        active_src = ColumnDataSource(data=dict(
            ch=active_chs,
            x=[row["x"] for row in sh["active"]],
            y=[row["y"] for row in sh["active"]],
            color=[region_colors[assignment[ch]] for ch in active_chs],
            line_color=[TRANSPARENT] * len(active_chs),
            line_width=[0] * len(active_chs),
        ))
        inactive_src = ColumnDataSource(data=dict(
            x=[row["x"] for row in sh["inactive"]],
            y=[row["y"] for row in sh["inactive"]],
        ))
        box_src = ColumnDataSource(data=dict(
            left=[b[0] for b in sh["boxes"]], bottom=[b[1] for b in sh["boxes"]],
            right=[b[2] for b in sh["boxes"]], top=[b[3] for b in sh["boxes"]],
        ))
        div_src = ColumnDataSource(data=dict(
            x0=[d[0] for d in sh["dividers"]], x1=[d[1] for d in sh["dividers"]],
            y0=[d[2] for d in sh["dividers"]], y1=[d[3] for d in sh["dividers"]],
        ))
        label_left_src = ColumnDataSource(data=dict(
            x=[l[0] for l in sh["labels_left"]], y=[l[1] for l in sh["labels_left"]],
            text=[l[2] for l in sh["labels_left"]],
        ))
        label_right_src = ColumnDataSource(data=dict(
            x=[l[0] for l in sh["labels_right"]], y=[l[1] for l in sh["labels_right"]],
            text=[l[2] for l in sh["labels_right"]],
        ))

        fig = figure(
            width=230, height=820, title=f"Shank {shank_id}",
            x_range=sh["x_range"], y_range=y_range,
            tools="", background_fill_color="#f5f5f5",
        )
        # dimensions="height": scroll-zoom only rescales y, never x -- matches
        # what the other (non-hovered) shanks already show, since only y_range
        # is shared across figures and x_range is per-figure.
        fig.add_tools(WheelZoomTool(dimensions="height"), PanTool(), ResetTool())
        fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)

        fig.quad(left="left", right="right", top="top", bottom="bottom", source=box_src,
                  fill_color=None, line_color="#444444", line_width=1.8)
        fig.segment(x0="x0", x1="x1", y0="y0", y1="y1", source=div_src,
                     line_color="#777777", line_width=0.7)
        fig.rect("x", "y", width=RECT_W, height=RECT_H, source=inactive_src,
                  fill_color=INACTIVE_COLOR, line_color=None)
        active_renderer = fig.rect(
            "x", "y", width=RECT_W, height=RECT_H, source=active_src,
            fill_color="color", line_color="line_color", line_width="line_width",
        )
        box_select = BoxSelectTool(renderers=[active_renderer])
        fig.add_tools(box_select)
        fig.toolbar.active_drag = box_select

        # click-to-toggle: a bare TapTool proved unreliable once gridplot's
        # merge_tools built a combined toolbar (its "active" state didn't
        # consistently carry over). A native Tap event handler with manual
        # nearest-channel hit-testing sidesteps that entirely.
        xs_arr  = np.array([row["x"] for row in sh["active"]], dtype=float)
        ys_arr  = np.array([row["y"] for row in sh["active"]], dtype=float)
        chs_arr = np.array(active_chs, dtype=int)

        def make_tap_handler(xs_arr=xs_arr, ys_arr=ys_arr, chs_arr=chs_arr):
            def handler(event):
                if len(chs_arr) == 0:
                    return
                dist2 = (xs_arr - event.x) ** 2 + (ys_arr - event.y) ** 2
                i = int(np.argmin(dist2))
                if dist2[i] > CLICK_RADIUS ** 2:
                    return
                ch = int(chs_arr[i])
                selection.symmetric_difference_update({ch})
                _refresh_highlight()
            return handler

        fig.on_event(Tap, make_tap_handler())

        fig.add_layout(LabelSet(
            x="x", y="y", text="text", source=label_left_src,
            text_align="right", text_font_size="8pt", text_font_style="bold",
            text_color="#333333", text_font="monospace",
        ))
        fig.add_layout(LabelSet(
            x="x", y="y", text="text", source=label_right_src,
            text_align="left", text_font_size="8pt", text_font_style="bold",
            text_color="#333333", text_font="monospace",
        ))

        fig.xaxis.axis_label = "xpos (µm)"
        if shank_id == 0:
            fig.yaxis.axis_label = "ypos (µm)"

        # selection callback: box-select is additive, single tap toggles ------
        def make_handler(active_src=active_src):
            def handler(attr, old, new):
                idx = list(new)
                if not idx:
                    return
                chs = active_src.data["ch"]
                if len(idx) == 1:
                    ch = int(chs[idx[0]])
                    selection.symmetric_difference_update({ch})
                else:
                    selection.update(int(chs[i]) for i in idx)
                active_src.selected.indices = []
                _refresh_highlight()
            return handler

        active_src.selected.on_change("indices", make_handler())
        active_sources.append(active_src)
        figs.append(fig)

    def _refresh_highlight():
        for src in active_sources:
            chs = src.data["ch"]
            src.data["line_color"] = [
                SELECT_EDGE_COLOR if int(ch) in selection else TRANSPARENT for ch in chs
            ]
            src.data["line_width"] = [
                SELECT_EDGE_WIDTH if int(ch) in selection else 0 for ch in chs
            ]

    grid = gridplot([figs], toolbar_location="above", merge_tools=True)
    # gridplot(merge_tools=True) builds a new combined toolbar for the layout,
    # which does not necessarily carry over the active_drag we set per-figure
    # above -- force it on the merged toolbar too, so box-select works without
    # the user having to click a toolbar icon first. (Click-to-toggle no longer
    # goes through a Tool at all -- see the Tap event handler above.)
    for tool in grid.toolbar.tools:
        if isinstance(tool, BoxSelectTool):
            grid.toolbar.active_drag = tool

    # control panel -------------------------------------------------------------
    # "unknown" isn't offered in the dropdown — it's reached via "Set to unknown"
    # below Assign, so it can't be picked by accident.
    selectable_regions = [r for r in BRAIN_REGIONS if r != "unknown"]
    region_select = pn.widgets.Select(name="Brain region", options=selectable_regions, width=200)
    btn_assign    = pn.widgets.Button(name="Assign", button_type="primary", width=200)
    btn_unassign  = pn.widgets.Button(name="Set to unknown", width=200)
    btn_clear     = pn.widgets.Button(name="Clear selection", width=200)
    btn_reset     = pn.widgets.Button(name="Reset all to unknown", button_type="danger", width=200)
    btn_save      = pn.widgets.Button(name="Save", button_type="success", width=200)
    status        = pn.pane.Markdown("", width=220)
    summary       = pn.pane.HTML("", width=220)

    # "Save" (while any channel is still "unknown") and "Reset all" both need a
    # second click to confirm; each tracked independently so one doesn't clear
    # the other's pending state when it's the one actually clicked.
    pending_confirm = {"save": False, "reset": False}

    def _reset_confirm(key):
        if not pending_confirm[key]:
            return
        pending_confirm[key] = False
        if key == "save":
            btn_save.name = "Save"
            btn_save.button_type = "success"
        else:
            btn_reset.name = "Reset all to unknown"
            btn_reset.button_type = "danger"

    def _apply_region(chs, region):
        chs = set(chs)
        for ch in chs:
            assignment[ch] = region
        for src in active_sources:
            data_chs = src.data["ch"]
            colors = list(src.data["color"])
            touched = False
            for i, ch in enumerate(data_chs):
                if int(ch) in chs:
                    colors[i] = region_colors[assignment[int(ch)]]
                    touched = True
            if touched:
                src.data["color"] = colors

    def _swatch(color):
        return (
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};margin-right:6px;border-radius:2px;'
            f'vertical-align:middle;"></span>'
        )

    def _update_summary():
        counts = {r: 0 for r in BRAIN_REGIONS}
        for r in assignment.values():
            counts[r] = counts.get(r, 0) + 1
        # "unknown" always shown first; other regions only if they have channels
        rows = [f'<div>{_swatch(region_colors["unknown"])}unknown: <b>{counts.get("unknown", 0)}</b></div>']
        for r in BRAIN_REGIONS:
            if r != "unknown" and counts.get(r, 0) > 0:
                rows.append(f'<div>{_swatch(region_colors[r])}{r}: <b>{counts[r]}</b></div>')
        summary.object = "<b>Assigned so far</b><br>" + "".join(rows)

    def on_assign(event):
        _reset_confirm("save")
        _reset_confirm("reset")
        if not selection:
            status.object = "No channels selected."
            return
        region = region_select.value
        changed = list(selection)
        selection.clear()
        _apply_region(changed, region)
        _refresh_highlight()
        _update_summary()
        status.object = f"Assigned **{region}** to {len(changed)} channel(s)."

    def on_unassign(event):
        _reset_confirm("save")
        _reset_confirm("reset")
        if not selection:
            status.object = "No channels selected."
            return
        changed = list(selection)
        selection.clear()
        _apply_region(changed, "unknown")
        _refresh_highlight()
        _update_summary()
        status.object = f"Set {len(changed)} channel(s) to 'unknown'."

    def on_clear(event):
        _reset_confirm("save")
        _reset_confirm("reset")
        selection.clear()
        _refresh_highlight()
        status.object = "Selection cleared."

    def on_reset(event):
        _reset_confirm("save")
        if not pending_confirm["reset"]:
            pending_confirm["reset"] = True
            btn_reset.name = "Confirm reset ALL"
            btn_reset.button_type = "warning"
            status.object = (
                f"This will set **all {len(channels)}** channels to 'unknown', "
                "including any already-labeled ones. Click **Confirm reset ALL** "
                "again to proceed."
            )
            return

        _reset_confirm("reset")
        selection.clear()
        _apply_region(channels, "unknown")
        _refresh_highlight()
        _update_summary()
        status.object = f"Reset all {len(channels)} channel(s) to 'unknown'."

    def on_save(event):
        _reset_confirm("reset")
        # trigger the confirm step on any channel still labeled "unknown"
        # (not just ones missing from `assignment`, since every channel is
        # defaulted to "unknown" from the start and would otherwise never trip)
        unknown_chs = [ch for ch in channels if assignment.get(ch, "unknown") == "unknown"]

        if unknown_chs and not pending_confirm["save"]:
            pending_confirm["save"] = True
            btn_save.name = "Confirm & Save"
            btn_save.button_type = "warning"
            status.object = (
                f"{len(unknown_chs)} channel(s) are still 'unknown' — click "
                "**Confirm & Save** again to save anyway."
            )
            return

        _reset_confirm("save")
        save_region_assignment(assignment, save_path)
        counts = {}
        for r in assignment.values():
            counts[r] = counts.get(r, 0) + 1
        summary_text = "\n".join(f"- {r}: {c}" for r, c in sorted(counts.items()))
        status.object = f"Saved to `{save_path.name}`\n\n{summary_text}"
        print(f"Saved region assignment ({len(assignment)} channels) -> {save_path}")
        for r, c in sorted(counts.items()):
            print(f"  {r:20s}: {c}")

    btn_assign.on_click(on_assign)
    btn_unassign.on_click(on_unassign)
    btn_clear.on_click(on_clear)
    btn_reset.on_click(on_reset)
    btn_save.on_click(on_save)

    _update_summary()

    controls = pn.Column(
        region_select, btn_assign, btn_unassign, btn_clear,
        pn.layout.Divider(), btn_reset, pn.layout.Divider(),
        btn_save, pn.layout.Divider(), summary, pn.layout.Divider(), status,
    )
    header = pn.pane.Markdown(f"### {title}" if title else "", margin=(0, 0, 0, 10))
    app = pn.Column(header, pn.Row(pn.pane.Bokeh(grid), controls))

    def get_assignment():
        return dict(assignment)

    return app, get_assignment
