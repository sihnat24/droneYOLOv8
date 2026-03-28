#!/usr/bin/env python3
"""
visualize.py — Diagnostic analysis of the exp9 (best) YOLOv8n VisDrone run.
Saves 5 PNGs to viz/ and prints a plain-text diagnostic summary.
"""
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
EXP = "runs/train/visdrone-nano-20260327_111438"
EXP_DIR      = Path(EXP)   # exp9
WEIGHTS      = EXP_DIR / "weights/best.pt"
RESULTS_CSV  = EXP_DIR / "results.csv"
DATASET_YAML = "dataset.yml"
VAL_IMAGES   = Path("datasets/yolo-VisDrone2019-DET/images/val")
VAL_LABELS   = Path("datasets/yolo-VisDrone2019-DET/labels/val")
VIZ_DIR      = Path("viz" + "-" + EXP)
VIZ_DIR.mkdir(exist_ok=True)

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor", "others",
]

random.seed(42)


# ── 1. Loss curves ───────────────────────────────────────────────────────────
def plot_loss_curves():
    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Loss Curves — Exp9  (close_mosaic=0  ·  lr0=0.02  ·  lrf=0.1)",
        fontsize=12,
    )

    for ax, loss in zip(axes, ["box_loss", "cls_loss", "dfl_loss"]):
        tr = df[f"train/{loss}"]
        va = df[f"val/{loss}"]
        ax.plot(df["epoch"], tr, "b-o", label="train", lw=2, markersize=5)
        ax.plot(df["epoch"], va, "r-o", label="val",   lw=2, markersize=5)
        ax.set_title(loss.replace("_loss", " loss"))
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
        if tr.iloc[-1] < tr.iloc[-3] and va.iloc[-1] < va.iloc[-3]:
            ax.annotate(
                "↘ both still declining — undertrained",
                xy=(0.04, 0.08), xycoords="axes fraction",
                fontsize=8, color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="darkred", alpha=0.8),
            )

    plt.tight_layout()
    out = VIZ_DIR / "1_loss_curves.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


# ── 2. Per-class recall & precision bar chart ────────────────────────────────
def plot_per_class(metrics):
    box   = metrics.box
    idxs  = np.array(box.ap_class_index)
    r     = np.array(box.r)
    p     = np.array(box.p)
    names = [CLASSES[i] for i in idxs]

    # sort by recall ascending so worst classes appear on the left
    order = np.argsort(r)
    r, p  = r[order], p[order]
    names = [names[i] for i in order]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, r, 0.38, label="recall",    color="steelblue", alpha=0.9)
    ax.bar(x + 0.2, p, 0.38, label="precision", color="tomato",    alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Per-class Recall & Precision  (sorted by recall ascending — worst left)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        f"worst recall: {names[0]}  ({r[0]:.3f})",
        xy=(0, r[0]), xytext=(0.5, 0.85), textcoords="axes fraction",
        fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred"),
    )

    plt.tight_layout()
    out = VIZ_DIR / "2_per_class.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")
    return names, r, p


# ── 3. PR curve ──────────────────────────────────────────────────────────────
def plot_pr_curve(metrics):
    box   = metrics.box
    r_pts = np.array(box.r)
    p_pts = np.array(box.p)

    fig, ax = plt.subplots(figsize=(10, 5))

    class_names = [CLASSES[i] for i in box.ap_class_index]
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    # Try smooth per-class curves from ultralytics internals
    try:
        px = np.linspace(0, 1, 1000)
        py = np.array(box.prec_values)   # [nc, 1000]
        for i in range(py.shape[0]):
            ax.plot(px, py[i], lw=0.8, alpha=0.5, color=colors[i], label=class_names[i])
        ax.plot(px, py.mean(0), "k-", lw=2, label="mean PR curve")
    except (AttributeError, TypeError):
        # fallback: per-class scatter
        for i, (r, p) in enumerate(zip(r_pts, p_pts)):
            ax.scatter([r], [p], s=60, color=colors[i], zorder=3, label=class_names[i])

    op_r = float(r_pts.mean())
    op_p = float(p_pts.mean())
    ax.scatter([op_r], [op_p], color="red", s=100, zorder=5, marker="*",
               label=f"mean op. point  P={op_p:.3f}  R={op_r:.3f}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve — Exp9")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = VIZ_DIR / "3_pr_curve.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


# ── 4. Prediction visualization — 2×4 grid ──────────────────────────────────
def plot_predictions(model):
    img_paths = sorted(VAL_IMAGES.glob("*.jpg"))
    chosen    = random.sample(img_paths, min(8, len(img_paths)))

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        "Ground truth (green) vs Predictions (red) — Exp9  [conf≥0.001]",
        fontsize=13,
    )

    for ax, img_path in zip(axes.flat, chosen):
        img  = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Load ground-truth YOLO labels
        gt_boxes = []
        lbl_path = VAL_LABELS / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw / 2) * W
                y1 = (yc - bh / 2) * H
                gt_boxes.append((x1, y1, bw * W, bh * H, int(cls)))

        # Run inference
        result = model.predict(
            str(img_path), device="mps", verbose=False, conf=0.001, max_det=300
        )[0]

        ax.imshow(img)
        for (x1, y1, bw, bh, _) in gt_boxes:
            ax.add_patch(patches.Rectangle(
                (x1, y1), bw, bh,
                linewidth=0.8, edgecolor="lime", facecolor="none",
            ))
        n_pred = 0
        if result.boxes is not None:
            n_pred = len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=0.8, edgecolor="red", facecolor="none",
                ))

        ax.set_title(
            f"{img_path.name[:22]}\n{len(gt_boxes)} GT  |  {n_pred} pred",
            fontsize=7,
        )
        ax.axis("off")

    plt.tight_layout()
    out = VIZ_DIR / "4_predictions.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


# ── 5. Confidence distribution ───────────────────────────────────────────────
def plot_confidence_dist(model, n_sample=200):
    img_paths = sorted(VAL_IMAGES.glob("*.jpg"))
    sample    = random.sample(img_paths, min(n_sample, len(img_paths)))

    all_confs = []
    for img_path in sample:
        result = model.predict(
            str(img_path), device="mps", verbose=False, conf=0.001, max_det=300
        )[0]
        if result.boxes is not None and len(result.boxes):
            all_confs.extend(result.boxes.conf.cpu().numpy().tolist())

    all_confs = np.array(all_confs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_confs, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0.25, color="red",    linestyle="--", lw=1.5, label="0.25 (typical threshold)")
    ax.axvline(0.50, color="orange", linestyle="--", lw=1.5, label="0.50")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Prediction Confidence Distribution — Exp9\n"
        f"{len(all_confs):,} detections from {len(sample)} sampled val images"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    pct_below_25 = (all_confs < 0.25).mean() * 100
    pct_below_50 = (all_confs < 0.50).mean() * 100
    ax.annotate(
        f"{pct_below_25:.0f}% < 0.25\n{pct_below_50:.0f}% < 0.50",
        xy=(0.06, 0.72), xycoords="axes fraction",
        fontsize=10, color="red",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", alpha=0.8),
    )

    plt.tight_layout()
    out = VIZ_DIR / "5_confidence_dist.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")
    return all_confs


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading exp9 weights…")
    model = YOLO(str(WEIGHTS))

    print("\n[1/5] Loss curves…")
    plot_loss_curves()

    print("[2-3/5] Validation (per-class metrics + PR curve)…")
    metrics = model.val(
        data=DATASET_YAML, device="mps", split="val", verbose=False, plots=False
    )
    plot_per_class(metrics)
    plot_pr_curve(metrics)

    print("[4/5] Prediction visualization (8 images)…")
    plot_predictions(model)

    print("[5/5] Confidence distribution (200 sampled val images)…")
    all_confs = plot_confidence_dist(model)

    # ── Diagnostic summary ────────────────────────────────────────────────────
    box         = metrics.box
    r_mean      = float(np.mean(box.r))
    p_mean      = float(np.mean(box.p))
    map50       = float(box.map50)
    pct_low     = float((all_confs < 0.25).mean() * 100)
    idxs        = np.array(box.ap_class_index)
    r_arr       = np.array(box.r)
    worst_idx   = int(np.argmin(r_arr))
    worst_class = CLASSES[idxs[worst_idx]]
    worst_r     = float(r_arr[worst_idx])



if __name__ == "__main__":
    main()
