#!/usr/bin/env python3
"""Quadrant-style layout for initial vs optimized MPO prompts.

- Top-left quadrant: everything for the initial (root) prompt — reference placeholder + instruction.
- Top-right quadrant: optimized reference image only.
- Bottom half (full width): optimized text instruction + image brief only.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


def load_image(path: Path):
    try:
        import matplotlib.image as mpimg

        return mpimg.imread(path)
    except Exception:
        from PIL import Image

        return Image.open(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.json.read_text())
    root = data["nodes_data"][0][0]
    best = data["train_best_node"]

    init_instr = root.get("instruction", "").strip()
    fin_instr = best.get("instruction", "").strip()
    init_img = root.get("mm_prompt_path")
    fin_img = best.get("mm_prompt_path")
    fin_cond = (best.get("mm_condition_prompt") or "").strip()

    root_tr = root.get("train_metric")
    best_tr = best.get("train_metric")
    test_m = best.get("test_metric") if isinstance(best.get("test_metric"), dict) else {}

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"MPO prompts & reference images — {args.json.name}\n"
        f"Initial (root id={root.get('id')}) vs final (best id={best.get('id')}, {best.get('action_type', '')})",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Row 0: two equal quadrants (initial | optimized image). Row 1: one band spanning both columns.
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.05, 0.95],
        width_ratios=[1, 1],
        hspace=0.14,
        wspace=0.12,
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.05,
    )

    ax_init = fig.add_subplot(gs[0, 0])
    ax_fin_img = fig.add_subplot(gs[0, 1])
    ax_fin_txt = fig.add_subplot(gs[1, :])

    # ---- Top-left: initial reference + instruction (entire quadrant) ----
    ax_init.set_xlim(0, 1)
    ax_init.set_ylim(0, 1)
    ax_init.axis("off")
    ax_init.set_title("Initial (root) — full multimodal prompt", fontsize=11, fontweight="bold", pad=8)

    # Sub-block: reference strip (~upper third of quadrant)
    if init_img and Path(init_img).is_file():
        ax_ir = ax_init.inset_axes([0.06, 0.55, 0.88, 0.38])
        ax_ir.imshow(load_image(Path(init_img)))
        ax_ir.axis("off")
        ax_ir.set_title("Reference image", fontsize=9, pad=2)
    else:
        patch = FancyBboxPatch(
            (0.06, 0.58),
            0.88,
            0.34,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            transform=ax_init.transAxes,
            facecolor="#e0e0e0",
            edgecolor="#888888",
            linewidth=1.2,
        )
        ax_init.add_patch(patch)
        ax_init.text(
            0.5,
            0.75,
            "No reference image\n(text-only multimodal prompt)",
            ha="center",
            va="center",
            fontsize=10,
            color="#333333",
            transform=ax_init.transAxes,
            linespacing=1.35,
        )

    init_meta = []
    if root_tr is not None:
        init_meta.append(f"Train acc. (full pass): {root_tr:.3f}")
    meta_line = (" · ".join(init_meta) + "\n\n") if init_meta else ""

    init_wrapped = textwrap.fill(init_instr, width=48, break_long_words=False, replace_whitespace=False)
    ax_init.text(
        0.06,
        0.52,
        meta_line + "Instruction:\n" + init_wrapped,
        ha="left",
        va="top",
        fontsize=9.5,
        transform=ax_init.transAxes,
        family="sans-serif",
    )

    # ---- Top-right: optimized image only ----
    ax_fin_img.axis("off")
    ax_fin_img.set_title("Optimized — reference image for the MLLM", fontsize=11, fontweight="bold", pad=8)
    if fin_img and Path(fin_img).is_file():
        ax_fin_img.imshow(load_image(Path(fin_img)))
    else:
        ax_fin_img.text(0.5, 0.5, "(missing image path)", ha="center", va="center", transform=ax_fin_img.transAxes)

    # ---- Bottom: full width — optimized instruction only ----
    ax_fin_txt.axis("off")
    ax_fin_txt.set_title(
        "Optimized — text instruction (+ image brief used to build the reference above)",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    hdr_bits = []
    if best_tr is not None:
        hdr_bits.append(f"train (posterior mean at end): {best_tr:.3f}")
    if test_m.get("acc") is not None:
        hdr_bits.append(f"test acc: {test_m['acc']:.3f}")
    if test_m.get("f1") is not None:
        hdr_bits.append(f"test F1 (macro): {test_m['f1']:.3f}")
    hdr = ("  |  ".join(hdr_bits) + "\n\n") if hdr_bits else ""

    fin_parts = [fin_instr]
    if fin_cond:
        fin_parts.append("\n\n— Image brief (sent to gpt-image) —\n" + fin_cond)
    fin_body = textwrap.fill("\n".join(fin_parts), width=118, break_long_words=False, replace_whitespace=False)

    ax_fin_txt.text(
        0.01,
        0.96,
        hdr + fin_body,
        transform=ax_fin_txt.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        family="sans-serif",
    )

    out = args.out or (args.json.parent / f"{args.json.stem}_prompts_images.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
