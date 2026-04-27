#!/usr/bin/env python3
"""Build a multi-panel summary figure from MPO cuckoo_*.json (+ optional log.log)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log_metrics(log_path: Path | None) -> dict:
    out = {
        "wall_time": None,
        "optim_cost_usd": None,
        "image_cost_usd": None,
        "n_image_logs": None,
        "n_model_output_logs": None,
    }
    if not log_path or not log_path.is_file():
        return out
    text = log_path.read_text(errors="replace")
    m = re.search(r"Done! Execution time: (\S+)", text)
    if m:
        out["wall_time"] = m.group(1)
    m = re.search(r"Optimizer Model:.*?Total cost: ([0-9.]+) USD", text)
    if m:
        out["optim_cost_usd"] = float(m.group(1))
    m = re.search(r"MM Generator Model:.*?Total cost: ([0-9.]+) USD", text)
    if m:
        out["image_cost_usd"] = float(m.group(1))
    out["n_image_logs"] = text.count("OPENAI IMAGE GENERATION COST:")
    out["n_model_output_logs"] = text.count("---------------\tModel Output\t----------------")
    return out


def collect_action_counts(total_nodes_data: list) -> dict[str, int]:
    counts = {"generation": 0, "edit": 0, "mix": 0, "root": 0}
    for batch in total_nodes_data:
        for n in batch:
            at = n.get("action_type")
            if at is None:
                counts["root"] += 1
            elif at in counts:
                counts[at] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True, help="cuckoo_N.json path")
    ap.add_argument("--log", type=Path, default=None, help="log.log for cost/time/proxies")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG (default: next to json)")
    args = ap.parse_args()

    data = json.loads(args.json.read_text())
    nodes_data = data["nodes_data"]
    total_nodes_data = data["total_nodes_data"]
    root_metric = nodes_data[0][0]["train_metric"]
    best = data["train_best_node"]
    best_train = best["train_metric"]
    test_m = best.get("test_metric") or {}
    test_acc = test_m.get("acc") if isinstance(test_m, dict) else None
    test_f1 = test_m.get("f1") if isinstance(test_m, dict) else None

    rounds = list(range(len(nodes_data)))
    beam_leader = [grp[0]["train_metric"] for grp in nodes_data]
    # Beam is sorted desc — leader should equal [0], verify with max
    best_so_far = []
    mx = root_metric
    for v in beam_leader:
        mx = max(mx, v)
        best_so_far.append(mx)

    cumulative_nodes = []
    s = 0
    for batch in total_nodes_data:
        s += len(batch)
        cumulative_nodes.append(s)

    actions = collect_action_counts(total_nodes_data)
    action_ops = {k: v for k, v in actions.items() if k != "root"}

    logm = parse_log_metrics(args.log)
    n_images_dir = None
    mpath = best.get("mm_prompt_path") or ""
    if "images" in str(mpath):
        img_dir = Path(mpath).parent
        if img_dir.is_dir():
            n_images_dir = len(list(img_dir.glob("*.jpg")))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"MPO run summary — {args.json.name}\n"
        f"Final train (best in beam): {best_train:.3f}  |  Test acc: {test_acc:.3f}  |  Test F1 (macro): {test_f1:.3f}",
        fontsize=11,
    )

    ax = axes[0, 0]
    ax.plot(rounds, beam_leader, "o-", label="Beam leader train acc.", color="#1f77b4")
    ax.plot(rounds, best_so_far, "--", label="Best-so-far (train)", color="#ff7f0e")
    ax.axhline(root_metric, color="gray", linestyle=":", label=f"Root baseline ({root_metric:.3f})")
    # Test is evaluated only once at the end (best train node); show as horizontal refs.
    if test_acc is not None:
        ax.axhline(
            test_acc,
            color="#d62728",
            linestyle="-.",
            linewidth=2,
            label=f"Test acc (final, best train) = {test_acc:.3f}",
        )
    if test_f1 is not None:
        ax.axhline(
            test_f1,
            color="#9467bd",
            linestyle="-.",
            linewidth=2,
            label=f"Test F1 macro (final) = {test_f1:.3f}",
        )
    ax.set_xlabel("Outer round (snapshot after beam update)")
    ax.set_ylabel("Accuracy / F1 (0–1 scale)")
    ax.set_title("Train vs round + final test (horizontal lines)\n(test = one shot on best train prompt)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    if test_acc is not None or test_f1 is not None:
        tlines = ["Final test (best train node):"]
        if test_acc is not None:
            tlines.append(f"  acc = {test_acc:.4f}")
        if test_f1 is not None:
            tlines.append(f"  F1 (macro) = {test_f1:.4f}")
        ax.text(
            0.03,
            0.92,
            "\n".join(tlines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5e6", edgecolor="#d62728", linewidth=1.2),
        )

    ax = axes[0, 1]
    ax.plot(rounds, cumulative_nodes, "s-", color="#2ca02c", markersize=4)
    ax.set_xlabel("Outer round")
    ax.set_ylabel("Cumulative candidates (nodes created)")
    ax.set_title("Search breadth (cumulative nodes)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    costs = []
    labels = []
    if logm["optim_cost_usd"] is not None:
        costs.append(logm["optim_cost_usd"])
        labels.append("GPT-4o-mini\n(optimizer)")
    if logm["image_cost_usd"] is not None:
        costs.append(logm["image_cost_usd"])
        labels.append("gpt-image-1")
    if costs:
        colors = ["#8c564b", "#e377c2"]
        ax.bar(labels, costs, color=colors[: len(costs)])
        ax.set_ylabel("USD (logged API estimate)")
        total = sum(costs)
        ax.set_title(f"OpenAI cost (total ≈ ${total:.2f})")
    else:
        ax.text(0.5, 0.5, "No cost lines in log\n(pass --log)", ha="center", va="center")
        ax.set_axis_off()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    names = list(action_ops.keys())
    vals = [action_ops[k] for k in names]
    ax.bar(names, vals, color=["#7fc97f", "#beaed4", "#fdc086"])
    ax.set_ylabel("Count")
    ax.set_title("New candidates by operator type")
    ax.grid(True, axis="y", alpha=0.3)

    lines = []
    if logm["wall_time"]:
        lines.append(f"Wall time: {logm['wall_time']}")
    if n_images_dir is not None:
        lines.append(f"Reference images (saved .jpg): {n_images_dir}")
    if logm["n_image_logs"] is not None:
        lines.append(f"gpt-image API calls (log lines): {logm['n_image_logs']}")
    if logm["n_model_output_logs"]:
        lines.append(f"Base-model eval blocks in log: {logm['n_model_output_logs']}")
    lines.append(f"Candidates total: {cumulative_nodes[-1]}")
    lines.append(f"Δ train vs root: {best_train - root_metric:+.3f}")

    fig.text(
        0.5,
        0.02,
        "  |  ".join(lines),
        ha="center",
        fontsize=9,
        style="italic",
        color="#333333",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = args.out or (args.json.parent / f"{args.json.stem}_summary.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
