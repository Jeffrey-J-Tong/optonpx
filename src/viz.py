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
