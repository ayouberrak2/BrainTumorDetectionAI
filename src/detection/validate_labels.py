from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def check_image_label_pairs(images_dir,labels_dir ):

    images_root = Path(images_dir)
    labels_root = Path(labels_dir)

    image_paths = [
        p for p in images_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_paths = list(labels_root.rglob("*.txt"))

    image_keys = {
        p.relative_to(images_root).with_suffix("").as_posix()
        for p in image_paths
    }

    label_keys = {
        p.relative_to(labels_root).with_suffix("").as_posix()
        for p in label_paths
    }

    missing_labels = sorted(image_keys - label_keys)
    orphan_labels = sorted(label_keys - image_keys)

    return {
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
    }