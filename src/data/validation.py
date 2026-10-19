
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


VALID_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".bmp",
    ".png",
}


def build_manifest(dataset_path) -> pd.DataFrame:
    root = Path(dataset_path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Le chemin n'est pas un dossier : {root}"
        )

    records = []

    class_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir()
    )

    if not class_dirs:
        raise ValueError(
            f"Aucun dossier de classe trouve dans : {root}"
        )

    for class_dir in class_dirs:
        label = class_dir.name
        class_count = 0


        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size

                if width <= 0 or height <= 0:
                    continue

                records.append({
                    "path": str(image_path.resolve()),
                    "label": label,
                    "extension": image_path.suffix.lower(),
                    "width": width,
                    "height": height,
                })

                class_count += 1

            except (
                OSError,
                ValueError,
                UnidentifiedImageError,
            ) as error:
                print(f"Image ignorée : {image_path} ({error})")


    columns = [
        "path",
        "label",
        "extension",
        "width",
        "height",
    ]

    manifest = pd.DataFrame(records, columns=columns)


    return manifest
