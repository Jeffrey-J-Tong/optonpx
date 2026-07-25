# Visualize agreement between KSLabel / Bombcell and manual Phy curation.
# Reads cluster_info.tsv from a Kilosort4 output directory.
#
# Environment: optonpx  (conda activate optonpx)

#%%
# ── paths ──────────────────────────────────────────────────────────────────────
from pathlib import Path

ks_dir = Path(
    r"E:\D1-1-4_IM-1971\ephys_raw\2026-06-19_14-59-14"
    r"\Record Node 101\experiment1\recording1\continuous"
    r"\OneBox-100.ProbeA\kilosort4_1_bombcell"
)

#%%
# ── load cluster_info ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MANUAL_LABELS = ["good", "mua", "noise"]
KS_LABELS     = ["good", "mua"]
BC_LABELS     = ["good", "mua", "noise"]
PALETTE       = {"good": "#2ecc71", "mua": "#3498db", "noise": "#e74c3c"}

df = pd.read_csv(ks_dir / "cluster_info.tsv", sep="\t")

# Normalise labels to lower-case for consistent comparison.
# Manual curation uses: good / mua / noise  (Phy default groups)
# KSLabel uses:         good / mua
# Bombcell uses:        GOOD / MUA / NOISE / NON-SOMA  (NON-SOMA treated as GOOD)
df["manual"]   = df["group"].str.lower().str.strip()
df["ks_label"] = df["KSLabel"].str.lower().str.strip()
df["bc_label"] = df["bc_unitType"].str.lower().str.strip().replace("non-soma", "good")

print(f"Total clusters: {len(df)}")
print("\nManual (Phy) label counts:")
print(df["manual"].value_counts().to_string())
print("\nKSLabel counts:")
print(df["ks_label"].value_counts().to_string())
print("\nBombcell label counts (non-soma → good):")
print(df["bc_label"].value_counts().to_string())


#%%
# ── helper: build confusion matrix ────────────────────────────────────────────
def _confusion(df, auto_col, auto_labels, manual_labels=None):
    """Return a DataFrame confusion matrix: rows=auto, cols=manual."""
    if manual_labels is None:
        manual_labels = sorted(df["manual"].unique())
    counts = (
        df.groupby([auto_col, "manual"])
          .size()
          .unstack(fill_value=0)
          .reindex(index=auto_labels, columns=manual_labels, fill_value=0)
    )
    return counts


#%%
# ── Fig 0: label distribution of each classifier ──────────────────────────────
def _plot_distribution(ax, series, labels, title):
    counts = series.value_counts().reindex(labels, fill_value=0)
    colors = [PALETTE.get(l, "#aaaaaa") for l in labels]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.7)
    total = counts.sum()
    for bar, v in zip(bars, counts.values):
        pct = 100 * v / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels([l.upper() for l in labels], fontsize=10)
    ax.set_ylabel("Number of units")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


fig0, axes0 = plt.subplots(1, 3, figsize=(13, 4))
fig0.suptitle("Label Distribution per Classifier", fontsize=13, fontweight="bold")

_plot_distribution(axes0[0], df["ks_label"], KS_LABELS,  f"KSLabel  (n={len(df)})")
_plot_distribution(axes0[1], df["bc_label"], BC_LABELS,  f"Bombcell  (n={len(df)})")
_plot_distribution(axes0[2], df["manual"],   MANUAL_LABELS, f"Manual (Phy)  (n={len(df)})")

fig0.tight_layout()

if "__file__" in dir():
    fig0.savefig(ks_dir / "label_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved label_distributions.png")
else:
    plt.show()


#%%
# ── Fig 1: KSLabel vs manual ──────────────────────────────────────────────────
cm_ks = _confusion(df, "ks_label", KS_LABELS, MANUAL_LABELS)
cm_bc = _confusion(df, "bc_label", BC_LABELS, MANUAL_LABELS)


def _plot_overlap(ax, cm, title, auto_labels, manual_labels):
    """Grouped bar chart: one group per auto label, bars = manual categories."""
    n_groups = len(auto_labels)
    n_bars   = len(manual_labels)
    x        = np.arange(n_groups)
    width    = 0.72 / n_bars

    for j, ml in enumerate(manual_labels):
        vals = [cm.loc[al, ml] if al in cm.index else 0 for al in auto_labels]
        offset = (j - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=PALETTE.get(ml, "#aaaaaa"), label=ml.capitalize(),
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(v), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in auto_labels], fontsize=10)
    ax.set_ylabel("Number of units")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Manual curation", framealpha=0.7)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("Automatic vs Manual Curation Agreement", fontsize=13, fontweight="bold")

_plot_overlap(axes1[0], cm_ks, "KSLabel vs Manual", KS_LABELS, MANUAL_LABELS)
_plot_overlap(axes1[1], cm_bc, "Bombcell vs Manual", BC_LABELS, MANUAL_LABELS)

fig1.tight_layout()

if "__file__" in dir():
    fig1.savefig(ks_dir / "curation_overlap.png", dpi=150, bbox_inches="tight")
    print("Saved curation_overlap.png")
else:
    plt.show()


#%%
# ── Fig 2: firing rate distribution of manually curated "good" units ──────────
good_df = df[df["manual"] == "good"].copy()

fig2, ax2 = plt.subplots(figsize=(7, 4))

fr = good_df["fr"].dropna()
bins = np.logspace(np.log10(max(fr.min(), 0.01)), np.log10(fr.max() + 0.1), 40)
ax2.hist(fr, bins=bins, color="#2ecc71", edgecolor="white", linewidth=0.5)

ax2.set_xscale("log")
ax2.set_xlabel("Firing rate (spikes / s)", fontsize=11)
ax2.set_ylabel("Number of units", fontsize=11)
ax2.set_title(
    f"Firing rate distribution — manually curated 'good' units  (n = {len(good_df)})",
    fontweight="bold",
)

# Annotate median
med = fr.median()
ax2.axvline(med, color="#27ae60", linestyle="--", linewidth=1.5,
            label=f"Median = {med:.1f} Hz")
ax2.legend(framealpha=0.7)
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))

fig2.tight_layout()

if "__file__" in dir():
    fig2.savefig(ks_dir / "good_units_fr_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved good_units_fr_distribution.png")
else:
    plt.show()

print(f"\nGood units: {len(good_df)}")
print(f"  FR median : {med:.2f} Hz")
print(f"  FR mean   : {fr.mean():.2f} Hz")
print(f"  FR range  : {fr.min():.2f} – {fr.max():.2f} Hz")
