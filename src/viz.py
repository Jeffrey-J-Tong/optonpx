import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from src.probe import (
    N_SHANKS, N_BANKS, N_SECTIONS, ROW_PITCH, ELECTRODE_COL_XS, ELEC_PER_SHANK,
    build_npx2_multishank_channel_mapping, build_npx2_multishank_channel_structure,
)

_mapping   = build_npx2_multishank_channel_mapping()
_structure = build_npx2_multishank_channel_structure(_mapping)

# Inverse of _mapping: inv_map[shank][bank][local_electrode] = channel — the
# real hardware channel number that reads a given electrode when that bank is
# selected on that shank. Static per-probe-type table (from probe.py's
# spreadsheet), independent of any specific recording's actual bank choice.
_inv_map = [[{} for _ in range(N_BANKS)] for _ in range(N_SHANKS)]
for _sh in range(N_SHANKS):
    for _bk in range(N_BANKS):
        for _ch in range(N_SECTIONS * 48):
            _e = _mapping[_sh][_ch][_bk]
            if _e is not None:
                _inv_map[_sh][_bk][_e] = _ch

BANK_LETTER_TO_INT = {"A": 0, "B": 1, "C": 2, "D": 3}

# Matches Open Ephys's own probe-view "Amplitude scale" legend (navy -> purple
# -> magenta -> coral -> orange -> yellow), which is more visually distinct
# than a stock matplotlib colormap like viridis/plasma for spotting per-electrode
# amplitude differences at a glance.
OPENEPHYS_AMPLITUDE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "openephys_amplitude",
    ["#1B0B78", "#6A0DAD", "#C2178C", "#F3776E", "#F7941D", "#FFF200"],
)


def plot_electrode_map(channel_electrode=None, title=None):
    """
    Draw a channel map for one NP2.0 multi-shank probe.

    Parameters
    ----------
    channel_electrode : dict {ch: global_electrode_index} or None
        probe["channel_electrode"] from probes_params.
        If None, draws the full probe layout with all electrodes in gray
        (no active channels highlighted).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    mapping   = _mapping
    structure = _structure

    BG, AX_BG, SPINE_COL = "white", "#f5f5f5", "#cccccc"
    RECT_W, RECT_H   = 10, 10
    BOX_X_PAD        = 16
    LABEL_MARGIN     = 22
    LABEL_INSET      = 3 * ROW_PITCH

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "wide_gradient",
        ["#6a1b9a", "#1565c0", "#0277bd", "#00838f",
         "#2e7d32", "#558b2f", "#f57f17", "#e65100", "#b71c1c"],
    )

    row_pitch = ROW_PITCH
    half_step = row_pitch / 2

    if channel_electrode is not None:
        channels = sorted(channel_electrode.keys())

        ch_shank = {}
        ch_bank  = {}
        for ch in channels:
            sh      = channel_electrode[ch] // ELEC_PER_SHANK
            local_e = channel_electrode[ch] % ELEC_PER_SHANK
            ch_shank[ch] = sh
            for bk in range(N_BANKS):
                if mapping[sh][ch][bk] == local_e:
                    ch_bank[ch] = bk
                    break

        active_banks  = sorted(set(ch_bank[ch] for ch in channels))
        banks_to_draw = list(range(max(active_banks) + 1))
        norm          = mcolors.Normalize(vmin=min(channels), vmax=max(channels))

        active_elec = set()
        for ch in channels:
            sh = ch_shank[ch]
            bk = ch_bank[ch]
            e  = mapping[sh][ch][bk]
            if e is not None:
                active_elec.add((sh, bk, e))
    else:
        channels      = []
        banks_to_draw = list(range(N_BANKS))
        norm          = mcolors.Normalize(vmin=0, vmax=N_SHANKS * ELEC_PER_SHANK - 1)
        active_elec   = set()

    drawn_ys = [
        (e // 2) * row_pitch
        for bk in banks_to_draw             # bk: bank
        for sh in range(N_SHANKS)           # sh: shank
        for sec in range(N_SECTIONS)        # sec: section
        for e in structure[sh][bk][sec]     # e: electrode
    ]
    y_pad = 25
    y_lim = (min(drawn_ys) - half_step - y_pad, max(drawn_ys) + half_step + y_pad)

    inv_map = [[{} for _ in range(N_BANKS)] for _ in range(N_SHANKS)]
    for sh in range(N_SHANKS):
        for bk in range(N_BANKS):
            for ch in range(N_SECTIONS * 48):
                e = mapping[sh][ch][bk]
                if e is not None:
                    inv_map[sh][bk][e] = ch

    sh_xs = ELECTRODE_COL_XS

    fig, axes = plt.subplots(
        1, N_SHANKS,
        figsize=(2.0 * N_SHANKS + 1.2, 13),
        sharey=True, constrained_layout=True,
    )
    fig.patch.set_facecolor(BG)

    for ax, shank_id in zip(axes, range(N_SHANKS)):
        ax.set_facecolor(AX_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)      # spines are the lines that bound the data area of a plot
        ax.tick_params(colors="#555555", labelsize=7)
        ax.set_xticks(sh_xs)

        box_x0 = min(sh_xs) - BOX_X_PAD
        box_x1 = max(sh_xs) + BOX_X_PAD
        box_w  = box_x1 - box_x0

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
            ax.add_patch(mpatches.Rectangle(
                (box_x0, box_y0), box_w, box_y1 - box_y0,
                linewidth=1.8, edgecolor="#444444", facecolor="none", zorder=2,
            ))

            sorted_secs = sorted(sec_y_ranges.items(), key=lambda t: t[1][0])
            for i in range(len(sorted_secs) - 1):
                y_div = (sorted_secs[i][1][1] + sorted_secs[i + 1][1][0]) / 2
                ax.plot([box_x0, box_x1], [y_div, y_div],
                        color="#777777", linewidth=0.7, zorder=2)

            for sec in range(N_SECTIONS):
                for e in structure[shank_id][bk][sec]:
                    ex = sh_xs[e % 2]
                    ey = (e // 2) * row_pitch
                    is_active = (shank_id, bk, e) in active_elec
                    color = cmap(norm(inv_map[shank_id][bk][e])) if is_active else "#cccccc"
                    ax.add_patch(mpatches.Rectangle(
                        (ex - RECT_W / 2, ey - RECT_H / 2), RECT_W, RECT_H,
                        facecolor=color, edgecolor="none", zorder=3,
                    ))

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
                    _, ch_bot = items[0]
                    _, ch_top = items[-1]
                    y_sec_bot = items[0][0]
                    y_sec_top = items[-1][0]
                    y_label_bot = y_sec_bot + LABEL_INSET
                    y_label_top = y_sec_top - LABEL_INSET
                    lx = box_x0 - 2 if parity == 0 else box_x1 + 2
                    ha = "right"    if parity == 0 else "left"
                    ax.text(lx, y_label_bot, str(ch_bot), fontsize=8, fontweight="bold",
                            ha=ha, va="center", fontfamily="monospace",
                            color="#333333", zorder=4)
                    ax.text(lx, y_label_top, str(ch_top), fontsize=8, fontweight="bold",
                            ha=ha, va="center", fontfamily="monospace",
                            color="#333333", zorder=4)

        ax.set_xlim(min(sh_xs) - BOX_X_PAD - LABEL_MARGIN, max(sh_xs) + BOX_X_PAD + LABEL_MARGIN)
        ax.set_ylim(*y_lim)
        ax.set_title(f"Shank {shank_id}", color="#333333", fontsize=9, pad=6)
        ax.set_xlabel("xpos (µm)", color="#555555", fontsize=11)

    axes[0].set_ylabel("ypos (µm)", color="#555555", fontsize=11)

    if channel_electrode is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.4, aspect=20, pad=0.02)
        cbar.set_ticks([min(channels), max(channels)])
        cbar.set_ticklabels([str(min(channels)), str(max(channels))])
        cbar.set_label("channel", color="#555555", fontsize=10)
        cbar.ax.tick_params(colors="#555555", labelsize=8)

    if title is not None:
        fig.suptitle(title, color="#222222", fontsize=11, fontweight="bold", y=1.01)

    return fig


def plot_probe_survey(electrodes, metric="peak_to_peak", title=None,
                      cmap=OPENEPHYS_AMPLITUDE_CMAP, vmin=0, vmax=500):
    """
    Draw a per-shank spatial heatmap of a Neuropixels probe-survey metric.

    Unlike plot_electrode_map (which highlights the channels selected for one
    recording), this draws every surveyed electrode on every bank, since a
    probe survey measures all of them — useful for picking which bank to
    select on each shank before recording.

    Parameters
    ----------
    electrodes : list[dict]
        One probe's "electrodes" list from an Open Ephys probe-survey JSON.
        Each dict must have "shank", "row", "column", "bank" (str "A".."D"),
        and the field named by `metric`.
    metric : str
        Key in each electrode dict to color by (default "peak_to_peak").
    title : str or None
    cmap : str or matplotlib.colors.Colormap
        Colormap for the continuous metric (default OPENEPHYS_AMPLITUDE_CMAP —
        navy/purple/magenta/coral/orange/yellow, matching Open Ephys's own
        probe-view "Amplitude scale" legend). Pass any matplotlib colormap
        name/object to use a different one.
    vmin, vmax : float or None
        Color scale range. Default 0/500 (µV), matching Open Ephys's default
        probe-view amplitude scale — pass None for either to use the data's
        own min/max instead.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from src.probe import ROW_PITCH, ELECTRODE_COL_XS

    BG, AX_BG, SPINE_COL = "white", "#f5f5f5", "#cccccc"
    RECT_W, RECT_H = 10, 10
    BOX_X_PAD = 16
    LABEL_MARGIN = 34    # reserved on BOTH sides, for channel-number + bank-letter text
    BANK_LABEL_OFFSET = 16   # extra left offset so the bank letter clears the channel-number column
    LABEL_INSET = 3 * ROW_PITCH

    banks_sorted = sorted(set(e["bank"] for e in electrodes))
    shanks_sorted = sorted(set(e["shank"] for e in electrodes))

    values = [e[metric] for e in electrodes]
    norm = mcolors.Normalize(
        vmin=vmin if vmin is not None else min(values),
        vmax=vmax if vmax is not None else max(values),
    )
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    by_shank = {sh: [] for sh in shanks_sorted}
    for e in electrodes:
        by_shank[e["shank"]].append(e)

    sh_xs = ELECTRODE_COL_XS
    all_ys = [e["row"] * ROW_PITCH for e in electrodes]
    y_pad = 25
    half_step = ROW_PITCH / 2
    y_lim = (min(all_ys) - half_step - y_pad, max(all_ys) + half_step + y_pad)

    # dedicated colorbar column so it never steals width from a shank subplot
    # (which would leave that shank narrower / off-center relative to the rest).
    # cax must NOT be in the shanks' sharey group -- sharing a colorbar axis's
    # y-scale with the (unrelated) µm y-range breaks constrained_layout's sizing.
    n_sh = len(shanks_sorted)
    fig = plt.figure(figsize=(2.0 * n_sh + 1.6, 13), constrained_layout=True)
    gs = fig.add_gridspec(1, n_sh + 1, width_ratios=[1] * n_sh + [0.12])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_sh)]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    cax = fig.add_subplot(gs[0, n_sh])
    fig.patch.set_facecolor(BG)

    box_x0 = min(sh_xs) - BOX_X_PAD
    box_x1 = max(sh_xs) + BOX_X_PAD
    box_w = box_x1 - box_x0
    x_lim = (box_x0 - LABEL_MARGIN, box_x1 + LABEL_MARGIN)   # symmetric -> box stays centered

    for i, (ax, sh) in enumerate(zip(axes, shanks_sorted)):
        ax.set_facecolor(AX_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)
        ax.tick_params(colors="#555555", labelsize=7)
        if i > 0:
            ax.tick_params(labelleft=False)   # sharey: avoid redundant y tick labels skewing widths
        ax.set_xticks(sh_xs)

        sh_electrodes = by_shank[sh]

        # draw electrodes, colored by metric
        for e in sh_electrodes:
            ex = sh_xs[e["column"]]
            ey = e["row"] * ROW_PITCH
            color = cmap_obj(norm(e[metric]))
            ax.add_patch(mpatches.Rectangle(
                (ex - RECT_W / 2, ey - half_step), RECT_W, RECT_H,
                facecolor=color, edgecolor="none", zorder=3,
            ))

        # bank boundary boxes, bank-letter labels, and real hardware channel
        # numbers (top/bottom channel per section, both column parities) —
        # channel numbers come from probe.py's electrode->channel table, i.e.
        # the channel that would read this electrode if this bank were
        # selected on this shank (same source plot_electrode_map/plot_imro use).
        for bank in banks_sorted:
            bk = BANK_LETTER_TO_INT[bank]
            bank_es = [e for e in sh_electrodes if e["bank"] == bank]
            if not bank_es:
                continue
            ys = [e["row"] * ROW_PITCH for e in bank_es]
            box_y0, box_y1 = min(ys) - half_step, max(ys) + half_step
            ax.add_patch(mpatches.Rectangle(
                (box_x0, box_y0), box_w, box_y1 - box_y0,
                linewidth=1.8, edgecolor="#444444", facecolor="none", zorder=2,
            ))
            ax.text(box_x0 - BANK_LABEL_OFFSET, (box_y0 + box_y1) / 2, bank, fontsize=9, fontweight="bold",
                    ha="right", va="center", fontfamily="monospace", color="#333333", zorder=4)

            for sec in range(N_SECTIONS):
                for parity in range(2):
                    items = []
                    for local_e in _structure[sh][bk][sec]:
                        if local_e % 2 != parity:
                            continue
                        ch = _inv_map[sh][bk].get(local_e)
                        if ch is None:
                            continue
                        items.append(((local_e // 2) * ROW_PITCH, ch))
                    if not items:
                        continue
                    items.sort()
                    y_label_bot = items[0][0] + LABEL_INSET
                    y_label_top = items[-1][0] - LABEL_INSET
                    lx = box_x0 - 2 if parity == 0 else box_x1 + 2
                    ha = "right" if parity == 0 else "left"
                    ax.text(lx, y_label_bot, str(items[0][1]), fontsize=6, fontweight="bold",
                            ha=ha, va="center", fontfamily="monospace", color="#333333", zorder=4)
                    ax.text(lx, y_label_top, str(items[-1][1]), fontsize=6, fontweight="bold",
                            ha=ha, va="center", fontfamily="monospace", color="#333333", zorder=4)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_title(f"Shank {sh}", color="#333333", fontsize=9, pad=6)
        ax.set_xlabel("xpos (µm)", color="#555555", fontsize=11)

    axes[0].set_ylabel("ypos (µm)", color="#555555", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(metric, color="#555555", fontsize=10)
    cbar.ax.tick_params(colors="#555555", labelsize=8)

    if title is not None:
        fig.suptitle(title, color="#222222", fontsize=11, fontweight="bold", y=1.01)

    return fig


def plot_probe_survey_bank_summary(electrodes, metric="peak_to_peak", title=None):
    """
    Grouped box plot of a probe-survey metric per (shank, bank) — a quick
    quantitative complement to plot_probe_survey for comparing banks.

    Parameters
    ----------
    electrodes : list[dict]
        One probe's "electrodes" list from an Open Ephys probe-survey JSON.
    metric : str
        Key in each electrode dict to summarize (default "peak_to_peak").
    title : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    shanks_sorted = sorted(set(e["shank"] for e in electrodes))
    banks_sorted = sorted(set(e["bank"] for e in electrodes))

    fig, ax = plt.subplots(figsize=(2.2 * len(shanks_sorted) + 1.5, 5))

    group_gap = 1.0
    box_width = 0.7
    positions, data, colors, tick_pos, tick_labels = [], [], [], [], []
    bank_cmap = plt.get_cmap("tab10")

    x = 0.0
    for sh in shanks_sorted:
        group_start = x
        for bi, bank in enumerate(banks_sorted):
            vals = [e[metric] for e in electrodes if e["shank"] == sh and e["bank"] == bank]
            if not vals:
                continue
            positions.append(x)
            data.append(vals)
            colors.append(bank_cmap(bi))
            x += 1.0
        tick_pos.append((group_start + x - 1.0) / 2)
        tick_labels.append(f"Shank {sh}")
        x += group_gap

    bp = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=True,
                    showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("#222222")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=9, color="#333333")
    ax.set_ylabel(metric, fontsize=10, color="#555555")
    ax.tick_params(axis="y", colors="#555555", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [mpatches.Patch(facecolor=bank_cmap(bi), alpha=0.7, label=bank)
              for bi, bank in enumerate(banks_sorted)]
    ax.legend(handles=handles, title="bank", fontsize=8, title_fontsize=8,
             loc="upper right", frameon=False)

    if title is not None:
        fig.suptitle(title, color="#222222", fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_probe_survey_interactive(electrodes, metric="peak_to_peak", title=None,
                                  cmap=OPENEPHYS_AMPLITUDE_CMAP, vmin=0, vmax=500,
                                  width=230, height=820):
    """
    Interactive (Bokeh) version of plot_probe_survey: same per-shank spatial
    layout and color scale, but hovering an electrode shows its identifying
    info — shank, bank, row/column, survey's global_index, the real hardware
    channel number (the channel that would read this electrode if this bank
    were selected on this shank — same lookup table plot_electrode_map/
    plot_imro use, from probe.py's spreadsheet; None where a slot has no
    corresponding channel, e.g. bank D's spare slots on shanks 0/1) — and
    the metric value.

    Parameters
    ----------
    electrodes, metric, title, cmap, vmin, vmax : see plot_probe_survey.
    width, height : int
        Size in pixels of each per-shank sub-figure.

    Returns
    -------
    grid : bokeh.layouts.gridplot
        Display it as the last expression of a notebook cell (after calling
        bokeh.io.output_notebook() once), or bokeh.plotting.show(grid).
    """
    from bokeh.io import output_notebook
    from bokeh.layouts import gridplot
    from bokeh.models import ColorBar, ColumnDataSource, HoverTool, Label, LinearColorMapper, Range1d
    from bokeh.plotting import figure

    from src.probe import ROW_PITCH, ELECTRODE_COL_XS

    output_notebook(hide_banner=True)

    BOX_X_PAD = 16
    LABEL_MARGIN = 30   # reserved on BOTH sides -> box stays centered in x_range
    RECT_W, RECT_H = 10, 10
    half_step = ROW_PITCH / 2
    sh_xs = ELECTRODE_COL_XS

    banks_sorted = sorted(set(e["bank"] for e in electrodes))
    shanks_sorted = sorted(set(e["shank"] for e in electrodes))

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    palette = [mcolors.to_hex(cmap_obj(i / 255)) for i in range(256)]
    values = [e[metric] for e in electrodes]
    lo = vmin if vmin is not None else min(values)
    hi = vmax if vmax is not None else max(values)
    color_mapper = LinearColorMapper(palette=palette, low=lo, high=hi)

    by_shank = {sh: [] for sh in shanks_sorted}
    for e in electrodes:
        by_shank[e["shank"]].append(e)

    all_ys = [e["row"] * ROW_PITCH for e in electrodes]
    y_pad = 25
    y_range = Range1d(min(all_ys) - half_step - y_pad, max(all_ys) + half_step + y_pad)
    box_x0 = min(sh_xs) - BOX_X_PAD
    box_x1 = max(sh_xs) + BOX_X_PAD
    x_range = (box_x0 - LABEL_MARGIN, box_x1 + LABEL_MARGIN)

    def _channel_for(e):
        bk = BANK_LETTER_TO_INT[e["bank"]]
        local_e = e["row"] * 2 + e["column"]
        return _inv_map[e["shank"]][bk].get(local_e)

    figs = []
    for sh in shanks_sorted:
        sh_electrodes = by_shank[sh]
        channels = [_channel_for(e) for e in sh_electrodes]
        src = ColumnDataSource(data=dict(
            x=[sh_xs[e["column"]] for e in sh_electrodes],
            y=[e["row"] * ROW_PITCH for e in sh_electrodes],
            value=[e[metric] for e in sh_electrodes],
            shank=[e["shank"] for e in sh_electrodes],
            bank=[e["bank"] for e in sh_electrodes],
            row=[e["row"] for e in sh_electrodes],
            column=[e["column"] for e in sh_electrodes],
            global_index=[e.get("global_index") for e in sh_electrodes],
            channel=[ch if ch is not None else "n/a" for ch in channels],
        ))

        fig = figure(
            width=width, height=height, title=f"Shank {sh}",
            x_range=x_range, y_range=y_range,
            tools="pan,wheel_zoom,reset", background_fill_color="#f5f5f5",
        )
        fig.yaxis.visible = sh == shanks_sorted[0]   # avoid redundant tick labels

        for bank in banks_sorted:
            bank_es = [e for e in sh_electrodes if e["bank"] == bank]
            if not bank_es:
                continue
            ys = [e["row"] * ROW_PITCH for e in bank_es]
            box_y0, box_y1 = min(ys) - half_step, max(ys) + half_step
            fig.quad(left=box_x0, right=box_x1, top=box_y1, bottom=box_y0,
                     fill_color=None, line_color="#444444", line_width=1.8)
            fig.add_layout(Label(
                x=box_x0 - 2, y=(box_y0 + box_y1) / 2, text=bank,
                text_align="right", text_baseline="middle",
                text_font_size="9pt", text_font_style="bold",
                text_color="#333333", text_font="monospace",
            ))

        renderer = fig.rect(
            "x", "y", width=RECT_W, height=RECT_H, source=src,
            fill_color={"field": "value", "transform": color_mapper}, line_color=None,
        )
        fig.add_tools(HoverTool(renderers=[renderer], tooltips=[
            ("channel", "@channel"),
            ("electrode", "@global_index"),
            ("shank / bank", "@shank / @bank"),
            ("row, col", "@row, @column"),
            (metric, "@value{0.0}"),
        ]))

        fig.xaxis.axis_label = "xpos (µm)"
        if sh == shanks_sorted[0]:
            fig.yaxis.axis_label = "ypos (µm)"
        figs.append(fig)

    # dedicated colorbar-only figure so it never steals width from a shank
    # subplot (which would leave that one narrower / off-center vs the rest)
    cbar_fig = figure(width=90, height=height, toolbar_location=None,
                      outline_line_color=None, background_fill_color=None)
    cbar_fig.axis.visible = False
    cbar_fig.grid.visible = False
    cbar_fig.add_layout(ColorBar(color_mapper=color_mapper, label_standoff=8, title=metric), "right")
    figs.append(cbar_fig)

    grid = gridplot([figs], toolbar_location="above")
    if title is not None:
        from bokeh.layouts import column
        from bokeh.models import Div
        header = Div(text=f"<b>{title}</b>", styles={"font-size": "13pt", "color": "#222222"})
        return column(header, grid)
    return grid
