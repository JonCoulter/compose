#!/usr/bin/env python3
"""
Build {task}_train.json / {task}_test.json + symlink images for MPO CUB tasks.

Expects official CUB-200-2011 extracted so that <cub_root>/CUB_200_2011/images exists.

Train/test totals are split across classes proportionally to folder size, disjoint images.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


def class_label_from_dirname(dirname: str) -> str:
    rest = dirname.split(".", 1)[1] if "." in dirname else dirname
    return rest.replace("_", " ")


def collect_classes(images_root: Path, min_images: int) -> list[Path]:
    dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    out = []
    for d in dirs:
        jpgs = sorted(d.glob("*.jpg"))
        if len(jpgs) >= min_images:
            out.append(d)
    return out


def allocate_integers(weights: list[int], total: int) -> list[int]:
    """Largest-remainder allocation of `total` across buckets with proportional weights."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return [0] * len(weights)
    s = sum(weights)
    if s == 0:
        raise ValueError("weights sum to zero")
    raw = [total * w / s for w in weights]
    floors = [int(x) for x in raw]
    rem = total - sum(floors)
    frac_idx = sorted(range(len(weights)), key=lambda i: raw[i] - floors[i], reverse=True)
    for k in range(rem):
        floors[frac_idx[k]] += 1
    return floors


def split_train_test_per_class(counts: list[int], train_total: int, test_total: int) -> tuple[list[int], list[int]]:
    """
    Return (train_per_class, test_per_class) with train_i + test_i <= counts[i],
    sum(train) == train_total, sum(test) == test_total.
    """
    if train_total + test_total > sum(counts):
        raise ValueError(
            f"train_total + test_total = {train_total + test_total} exceeds "
            f"available images {sum(counts)} across selected classes."
        )
    if train_total > sum(counts) or test_total > sum(counts):
        raise ValueError("train_total or test_total larger than total pool.")

    trains = allocate_integers(counts, train_total)
    caps_remain = [c - t for c, t in zip(counts, trains)]
    if test_total > sum(caps_remain):
        raise ValueError(
            f"test_total={test_total} impossible after train split; "
            f"only {sum(caps_remain)} images left."
        )
    tests = allocate_integers(caps_remain, test_total)
    for i, (t, e, n) in enumerate(zip(trains, tests, counts)):
        if t + e > n:
            raise ValueError(f"class {i}: train+test {t}+{e} > {n}")
    return trains, tests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cub-root", type=Path, required=True)
    p.add_argument("--mpo-data-dir", type=Path, required=True)
    p.add_argument("--task", type=str, default="cuckoo")
    p.add_argument(
        "--train-total",
        type=int,
        default=None,
        help="Total training rows (split across classes). Use with --test-total.",
    )
    p.add_argument(
        "--test-total",
        type=int,
        default=None,
        help="Total test rows (split across classes). Use with --train-total.",
    )
    p.add_argument(
        "--train-per-class",
        type=int,
        default=None,
        help="Legacy: fixed count per class (ignored if --train-total set).",
    )
    p.add_argument(
        "--test-per-class",
        type=int,
        default=None,
        help="Legacy: fixed count per class (ignored if --train-total set).",
    )
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefer-keyword", type=str, default="Cuckoo")
    args = p.parse_args()

    if (args.train_total is None) != (args.test_total is None):
        p.error("Provide both --train-total and --test-total, or neither (then use per-class counts).")
    use_totals = args.train_total is not None
    if not use_totals:
        if args.train_per_class is None:
            args.train_per_class = 4
        if args.test_per_class is None:
            args.test_per_class = 2
    else:
        if args.train_per_class is not None or args.test_per_class is not None:
            print("Note: --train-total/--test-total take precedence over per-class flags.", file=sys.stderr)

    cub_images = args.cub_root / "CUB_200_2011" / "images"
    if not cub_images.is_dir():
        raise SystemExit(f"Missing CUB images dir: {cub_images}")

    if use_totals:
        need_min = 1
    else:
        need_min = args.train_per_class + args.test_per_class

    candidates = collect_classes(cub_images, need_min)
    if len(candidates) < args.classes:
        raise SystemExit(f"Need {args.classes} classes with at least {need_min} images each; found {len(candidates)}")

    kw = args.prefer_keyword.lower()
    preferred = [c for c in candidates if kw in c.name.lower()]
    chosen: list[Path] = []
    for c in preferred:
        if len(chosen) >= args.classes:
            break
        chosen.append(c)
    for c in candidates:
        if len(chosen) >= args.classes:
            break
        if c not in chosen:
            chosen.append(c)

    class_counts = [len(sorted(d.glob("*.jpg"))) for d in chosen]
    sum_n = sum(class_counts)

    if use_totals:
        T, E = args.train_total, args.test_total
        if T + E > sum_n:
            print(
                f"ERROR: train_total + test_total = {T + E} > {sum_n} images available "
                f"in these {args.classes} classes ({class_counts}). "
                f"Maximum disjoint split uses all {sum_n} images (e.g. train={sum_n//2}, test={sum_n - sum_n//2}).",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            train_counts, test_counts = split_train_test_per_class(class_counts, T, E)
        except ValueError as e:
            raise SystemExit(str(e))
    else:
        tpc, epc = args.train_per_class, args.test_per_class
        train_counts = [tpc] * args.classes
        test_counts = [epc] * args.classes
        for i, n in enumerate(class_counts):
            if train_counts[i] + test_counts[i] > n:
                raise SystemExit(
                    f"class {chosen[i].name}: need {tpc}+{epc} images but only {n} available"
                )

    mpo_cub = args.mpo_data_dir / "classification" / "cub"
    mpo_images = mpo_cub / "images"
    mpo_cub.mkdir(parents=True, exist_ok=True)

    if mpo_images.is_symlink() or mpo_images.is_dir():
        if mpo_images.is_symlink() or mpo_images.is_file():
            mpo_images.unlink()
        else:
            shutil.rmtree(mpo_images)
    mpo_images.symlink_to(cub_images.resolve(), target_is_directory=True)

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    rng = random.Random(args.seed)

    for class_dir, label, nt, ne in zip(chosen, [class_label_from_dirname(d.name) for d in chosen], train_counts, test_counts):
        jpgs = sorted(class_dir.glob("*.jpg"))
        rng.shuffle(jpgs)
        need = nt + ne
        if len(jpgs) < need:
            raise SystemExit(f"{class_dir.name}: need {need} images, have {len(jpgs)}")
        rel = lambda path: f"{path.parent.name}/{path.name}"
        for img in jpgs[:nt]:
            train_rows.append({"filename": rel(img), "label": label})
        for img in jpgs[nt : nt + ne]:
            test_rows.append({"filename": rel(img), "label": label})

    train_path = mpo_cub / f"{args.task}_train.json"
    test_path = mpo_cub / f"{args.task}_test.json"
    with open(train_path, "w") as f:
        json.dump(train_rows, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_rows, f, indent=2)

    print(f"Wrote {train_path} ({len(train_rows)} rows)")
    print(f"Wrote {test_path} ({len(test_rows)} rows)")
    print(f"Per-class train counts: {train_counts}, test counts: {test_counts}")
    print(f"Symlink {mpo_images} -> {cub_images.resolve()}")
    print("Classes:", [class_label_from_dirname(d.name) for d in chosen])


if __name__ == "__main__":
    main()
