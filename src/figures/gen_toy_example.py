"""
Generate a toy example figure comparing uniform spectral routing (ACM-GNN)
with per-edge cost routing (CSNA).

Scenario: A small heterophilous graph where cross-class edges have mixed utility.
Some heterophilous edges carry complementary information (helpful) while others
are misleading (harmful). Uniform channel treatment cannot distinguish them.

Saves to: paper/figures/toy_example.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Graph definition ──────────────────────────────────────────────────────────
# 8 nodes, 2 classes (A=blue, B=red)
# Positions laid out by hand for visual clarity
node_class = {0: "A", 1: "A", 2: "A", 3: "A",
              4: "B", 5: "B", 6: "B", 7: "B"}

# Positions (x, y) – arranged so classes are somewhat mixed
pos = {
    0: (0.8, 2.5),
    1: (2.2, 3.2),
    2: (1.0, 0.8),
    3: (2.8, 1.5),
    4: (1.8, 1.8),  # central B node
    5: (3.5, 2.8),
    6: (0.2, 1.6),
    7: (3.2, 0.3),
}

# Edge list: (src, dst, edge_type)
# edge_type: "homo" = same-class, "helpful_hetero" = cross-class but complementary,
#            "harmful_hetero" = cross-class and misleading
edges = [
    (0, 1, "homo"),              # A-A
    (0, 4, "helpful_hetero"),    # A-B helpful
    (1, 5, "harmful_hetero"),    # A-B harmful
    (2, 4, "harmful_hetero"),    # A-B harmful
    (2, 6, "helpful_hetero"),    # A-B helpful (cross-class, but 6 is B near A cluster)
    (3, 7, "helpful_hetero"),    # A-B helpful
    (3, 4, "harmful_hetero"),    # A-B harmful
    (4, 5, "homo"),              # B-B
    (5, 7, "homo"),              # B-B
    (1, 4, "helpful_hetero"),    # A-B helpful
    (6, 0, "harmful_hetero"),    # B-A harmful
]

# ── Colors ────────────────────────────────────────────────────────────────────
CLASS_A_COLOR = "#4393E5"   # blue
CLASS_B_COLOR = "#E5533E"   # red
HOMO_COLOR = "#888888"
HELPFUL_COLOR = "#2CA02C"   # green
HARMFUL_COLOR = "#D62728"   # dark red
BG_COLOR = "#FAFAFA"

def get_node_color(node_id):
    return CLASS_A_COLOR if node_class[node_id] == "A" else CLASS_B_COLOR

def draw_panel(ax, title, subtitle, edge_weights, show_legend=False):
    """Draw one panel of the comparison figure."""
    ax.set_xlim(-0.5, 4.2)
    ax.set_ylim(-0.3, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.text(1.85, 3.85, subtitle, fontsize=8.5, ha="center", va="top",
            fontstyle="italic", color="#555555")

    # Draw edges
    for (u, v, etype), (width, color, alpha, style) in zip(edges, edge_weights):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=width,
                alpha=alpha, linestyle=style, solid_capstyle="round",
                zorder=1)

    # Draw nodes
    for nid, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.18, color=get_node_color(nid),
                             ec="white", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(nid), fontsize=8, ha="center", va="center",
                color="white", fontweight="bold", zorder=4)

    if show_legend:
        # Build legend handles for the right panel
        handles = [
            mpatches.Patch(color=CLASS_A_COLOR, label="Class A"),
            mpatches.Patch(color=CLASS_B_COLOR, label="Class B"),
            plt.Line2D([0], [0], color=HELPFUL_COLOR, linewidth=2.5,
                       label="Helpful cross-class (upweighted)"),
            plt.Line2D([0], [0], color=HARMFUL_COLOR, linewidth=1.0,
                       alpha=0.4, linestyle="--",
                       label="Harmful cross-class (downweighted)"),
            plt.Line2D([0], [0], color=HOMO_COLOR, linewidth=1.8,
                       label="Same-class"),
        ]
        ax.legend(handles=handles, loc="lower center", fontsize=7,
                  ncol=2, framealpha=0.9, edgecolor="#CCCCCC",
                  bbox_to_anchor=(0.5, -0.08))


# ── Build edge weight specs for each panel ────────────────────────────────────
# ACM-GNN panel: uniform treatment per spectral band
# All heterophilous edges get the same high-pass weight; all homophilous get low-pass
uniform_weights = []
for u, v, etype in edges:
    if etype == "homo":
        uniform_weights.append((1.8, HOMO_COLOR, 0.7, "-"))
    else:
        # Uniform high-pass: same thickness for ALL cross-class edges
        uniform_weights.append((2.2, "#9467BD", 0.65, "-"))  # purple = uniform channel

# CSNA panel: per-edge routing distinguishes helpful from harmful
csna_weights = []
for u, v, etype in edges:
    if etype == "homo":
        csna_weights.append((1.8, HOMO_COLOR, 0.7, "-"))
    elif etype == "helpful_hetero":
        csna_weights.append((3.2, HELPFUL_COLOR, 0.85, "-"))    # thick = upweighted
    else:  # harmful_hetero
        csna_weights.append((0.8, HARMFUL_COLOR, 0.35, "--"))   # thin dashed = downweighted

# ── Create figure ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0),
                                facecolor="white")
fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.88, bottom=0.10)

draw_panel(ax1,
           "Uniform spectral routing (ACM-GNN)",
           "All cross-class edges receive identical high-pass weight",
           uniform_weights, show_legend=False)

draw_panel(ax2,
           "Per-edge cost routing (CSNA)",
           "Each edge receives an individualized routing weight",
           csna_weights, show_legend=True)

# Add legend for left panel
handles_left = [
    mpatches.Patch(color=CLASS_A_COLOR, label="Class A"),
    mpatches.Patch(color=CLASS_B_COLOR, label="Class B"),
    plt.Line2D([0], [0], color="#9467BD", linewidth=2.2,
               label="Uniform high-pass channel"),
    plt.Line2D([0], [0], color=HOMO_COLOR, linewidth=1.8,
               label="Same-class"),
]
ax1.legend(handles=handles_left, loc="lower center", fontsize=7,
           ncol=2, framealpha=0.9, edgecolor="#CCCCCC",
           bbox_to_anchor=(0.5, -0.08))

out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "toy_example.pdf")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
