import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from collections import Counter
from hafnia.dataset.hafnia_dataset import HafniaDataset

dataset = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")
samples = dataset.samples

CLASS_COLORS = {
    "Vehicle.Car":              "#e6194B",
    "Vehicle.Pickup Truck":     "#f58231",
    "Vehicle.Single Truck":     "#ffe119",
    "Vehicle.Combo Truck":      "#bfef45",
    "Vehicle.Heavy Duty Vehicle": "#3cb44b",
    "Vehicle.Trailer":          "#42d4f4",
    "Vehicle.Motorcycle":       "#4363d8",
    "Vehicle.Bicycle":          "#911eb4",
    "Vehicle.Van":              "#f032e6",
    "Person":                   "#a9a9a9",
}

# --- 1. Examples with bounding boxes ---
annotated = [r for r in samples.iter_rows(named=True) if r["bboxes"]]
random.seed(7)
picked = random.sample(annotated, 9)

fig, axes = plt.subplots(3, 3, figsize=(18, 11))
for ax, row in zip(axes.flat, picked):
    img = Image.open(row["file_path"])
    W, H = img.size
    ax.imshow(img)
    for b in row["bboxes"]:
        x = b["top_left_x"] * W
        y = b["top_left_y"] * H
        w = b["width"] * W
        h = b["height"] * H
        cls = b["class_name"]
        color = CLASS_COLORS.get(cls, "white")
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x, max(y - 4, 0), cls.replace("Vehicle.", ""),
                color="black", fontsize=7,
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.85, pad=1))
    ax.set_title(f"split={row['split']}  bboxes={len(row['bboxes'])}", fontsize=9)
    ax.axis("off")

fig.suptitle("ECCV Cross-City — sample images with bounding boxes", fontsize=14)
fig.tight_layout()
fig.savefig("/home/dsa/hafnia/visualization/examples.png", dpi=110, bbox_inches="tight")
print("Saved examples.png")

# --- 2. Class distribution bar chart ---
counts = Counter()
for row in samples.iter_rows(named=True):
    for b in row["bboxes"] or []:
        counts[b["class_name"]] += 1

ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
labels = [c.replace("Vehicle.", "") for c, _ in ordered]
values = [v for _, v in ordered]
colors = [CLASS_COLORS.get(c, "#888") for c, _ in ordered]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.4)
ax.set_ylabel("number of bbox annotations")
ax.set_title(f"Class distribution — {sum(values)} bboxes over {len(samples)} samples")
ax.set_yscale("log")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=25, ha="right")
fig.tight_layout()
fig.savefig("/home/dsa/hafnia/visualization/class_distribution.png", dpi=120, bbox_inches="tight")
print("Saved class_distribution.png")

# --- 3. Per-split bbox count ---
split_counts = Counter()
split_samples = Counter()
for row in samples.iter_rows(named=True):
    split_samples[row["split"]] += 1
    for _ in row["bboxes"] or []:
        split_counts[row["split"]] += 1

splits = ["train", "validation", "test"]
fig, ax = plt.subplots(figsize=(7, 4))
xs = range(len(splits))
ax.bar([x - 0.2 for x in xs], [split_samples[s] for s in splits], width=0.4, label="samples", color="#4363d8")
ax.bar([x + 0.2 for x in xs], [split_counts[s] for s in splits], width=0.4, label="bboxes", color="#e6194B")
ax.set_xticks(list(xs))
ax.set_xticklabels(splits)
ax.set_ylabel("count")
ax.set_title("Samples and bbox counts per split")
for x, s in zip(xs, splits):
    ax.text(x - 0.2, split_samples[s], str(split_samples[s]), ha="center", va="bottom", fontsize=9)
    ax.text(x + 0.2, split_counts[s], str(split_counts[s]), ha="center", va="bottom", fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig("/home/dsa/hafnia/visualization/splits.png", dpi=120, bbox_inches="tight")
print("Saved splits.png")
