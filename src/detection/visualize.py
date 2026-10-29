"""Afficher des images IRM et des bounding boxes."""
from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.detection.validate_labels import read_yolo_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def show_images(images_dir, labels_dir, class_names, n=6, seed=42):
    """Afficher un échantillon depuis data/processed/detection/<split>."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print(f"Aucune image dans {images_dir}")
        return

    chosen = random.Random(seed).sample(images, min(n, len(images)))
    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4))
    if len(chosen) == 1:
        axes = [axes]

    for ax, image_path in zip(axes, chosen):
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        width, height = image.size
        ax.imshow(image)
        label_path = labels_dir / f"{image_path.stem}.txt"
        objects = read_yolo_label(label_path) if label_path.exists() else []

        for class_id, x, y, w, h in objects:
            left = (x - w / 2) * width
            top = (y - h / 2) * height
            ax.add_patch(Rectangle((left, top), w * width, h * height,
                                   fill=False, edgecolor="lime", linewidth=2))
            name = class_names[class_id]
            ax.text(left, max(0, top - 3), name, color="yellow", fontsize=10,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
        ax.set_title(image_path.name[:32], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
