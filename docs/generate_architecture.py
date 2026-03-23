"""Generate the AgentOS architecture diagram using matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
PNG_PATH = ASSETS / "architecture.png"
SVG_PATH = ASSETS / "architecture.svg"


def _draw_box(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str = "") -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#3B4254",
        facecolor="#171B26",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.63,
        title,
        ha="center",
        va="center",
        color="#E6EDF7",
        fontsize=12,
        fontweight="bold",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.34,
            subtitle,
            ha="center",
            va="center",
            color="#9FB1CC",
            fontsize=9.5,
        )


def generate() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9), dpi=220)
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.965,
        "AgentOS System Architecture",
        ha="center",
        va="center",
        color="#E6EDF7",
        fontsize=18,
        fontweight="bold",
    )

    # Layer coordinates
    left = 0.08
    width = 0.84
    h = 0.105
    gap = 0.025
    y = 0.82

    # Layer 1
    _draw_box(ax, left, y, width, h, "AgentOS CLI", "agentos serve / init / mcp")
    y -= h + gap

    # Layer 2
    _draw_box(
        ax,
        left,
        y,
        width,
        h,
        "Web Platform (FastAPI)",
        "Agent Builder | Chat | Dashboard | Embed",
    )
    y -= h + gap

    # Layer 3 (4 modules)
    mod_w = (width - 3 * 0.012) / 4
    x = left
    layer3 = [
        ("Agent SDK", ""),
        ("Sandbox", "Testing"),
        ("Monitor", "Events"),
        ("Governance", "Budget + Auth"),
    ]
    for t, s in layer3:
        _draw_box(ax, x, y, mod_w, h, t, s)
        x += mod_w + 0.012
    y -= h + gap

    # Layer 4
    _draw_box(
        ax,
        left,
        y,
        width,
        h,
        "Core Engine",
        "Tool Calling | Streaming | Memory | RAG",
    )
    y -= h + gap

    # Layer 5 (providers)
    x = left
    layer5 = [
        ("OpenAI", "Provider"),
        ("Claude", "Provider"),
        ("Ollama", "Provider"),
        ("Mock", "Provider"),
    ]
    for t, s in layer5:
        _draw_box(ax, x, y, mod_w, h, t, s)
        x += mod_w + 0.012
    y -= h + gap

    # Layer 6
    _draw_box(ax, left, y, width, h, "MCP Server (JSON-RPC/stdio)", "")

    # Accent separators
    sep_color = "#2F81F7"
    for y_sep in [0.81, 0.68, 0.55, 0.42, 0.29, 0.16]:
        ax.plot([left, left + width], [y_sep, y_sep], color=sep_color, alpha=0.15, lw=1)

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    generate()
