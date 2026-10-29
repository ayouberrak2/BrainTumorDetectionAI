"""Préparer les IRM pour la détection YOLOv8.

Structure d'entrée :
  data/raw/detection/Train/Glioma/images/a.jpg
  data/raw/detection/Train/Glioma/labels/a.txt
  ... même structure pour Meningioma, No Tumor, Pituitary et Val/Test.

Deux modes possibles :
- Par défaut : un seul ID brut est autorisé par dossier de tumeur.
- Avec source_id_to_class explicite : les IDs sont interprétés via le
  dictionnaire fourni, et NON à partir du dossier d'origine.

Avant de fournir un dictionnaire, vérifier data.yaml/classes.txt source.
Ne jamais inventer la correspondance des IDs.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import random
import shutil

from src.detection.validate_labels import read_yolo_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# No Tumor n'est pas une boîte ; c'est un exemple négatif.
CLASS_IDS = {"glioma": 0, "meningioma": 1, "pituitary": 2}
CLASS_NAMES = ["glioma", "meningioma", "pituitary"]


def normalize_folder(folder_name):
    return folder_name.lower().replace(" ", "").replace("_", "").replace("-", "")


def scan_dataset(raw_root):
    """Lister images, étiquettes manquantes et fichiers .txt isolés."""
    raw_root = Path(raw_root)
    rows = []
    for split_dir in sorted(raw_root.iterdir()):
        if not split_dir.is_dir() or split_dir.name.lower() not in {"train", "val", "valid", "validation", "test"}:
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            folder = normalize_folder(class_dir.name)
            if folder not in {"glioma", "meningioma", "notumor", "pituitary"}:
                print(f"Dossier ignoré (classe inconnue): {class_dir}")
                continue

            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if images_dir.exists() else []
            labels = sorted(p for p in labels_dir.glob("*.txt") if p.is_file()) if labels_dir.exists() else []
            labels_by_name = {p.stem.lower(): p for p in labels}
            image_names = {p.stem.lower() for p in images}

            for image in images:
                label = labels_by_name.get(image.stem.lower())
                rows.append({"split": split_dir.name, "class_folder": class_dir.name,
                             "image": str(image), "label": str(label) if label else "",
                             "status": "paired" if label else "missing_label"})
            for label in labels:
                if label.stem.lower() not in image_names:
                    rows.append({"split": split_dir.name, "class_folder": class_dir.name,
                                 "image": "", "label": str(label), "status": "orphan_label"})
    return rows


def _sha256(image_path):
    """Empreinte permettant de détecter les images strictement identiques."""
    sha = hashlib.sha256()
    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def prepare_dataset(raw_root, output_root, test_ratio=0.20, seed=42,
                    source_id_to_class=None, allow_cross_class_boxes=False):
    """Valider, répartir et copier images/labels au format YOLO.

    - Conserve Val comme validation ; conserve Test s'il existe.
    - S'il n'existe pas de Test, prélève 20% de Train par classe.
    - Fichiers manquants: ignorés, SAUF No Tumor (label vide créé).
    - Élimine les copies d'images identiques entre splits (SHA-256).
    - N'altère jamais les données brutes.
    - Si plusieurs IDs existent dans un dossier, spécifier source_id_to_class
      (ex. {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}),
      mais UNIQUEMENT après vérification des noms de classes source.
    - Par sécurité, ignore les boîtes dont la classe ne correspond pas
      au dossier. Lever la restriction seulement si les images mixtes
      ont été vérifiées : allow_cross_class_boxes=True.
    """
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Dossier brut introuvable : {raw_root}")
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("Le dossier de sortie ne doit pas être dans data/raw/detection.")
    if output_root.name != "detection":
        raise ValueError("Par sécurité, le dossier de sortie doit s'appeler 'detection'.")
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio doit être entre 0 et 1.")

    # Ne pas deviner les IDs source : ils peuvent différer d'un dataset à l'autre.
    if source_id_to_class is not None:
        if not isinstance(source_id_to_class, dict) or not source_id_to_class:
            raise ValueError("source_id_to_class doit être un dictionnaire non vide.")
        source_id_to_class = {int(k): normalize_folder(v)
                              for k, v in source_id_to_class.items()}
        unknown_classes = set(source_id_to_class.values()) - set(CLASS_IDS) - {"notumor"}
        if unknown_classes:
            raise ValueError(f"Classes inconnues dans le mapping : {unknown_classes}")

    scan_rows = scan_dataset(raw_root)
    if not any(r["split"].lower() == "train" for r in scan_rows):
        raise ValueError("Aucun jeu Train trouvé.")
    if not any(r["split"].lower() in {"val", "valid", "validation"} for r in scan_rows):
        raise ValueError("Aucun jeu Val trouvé.")

    reports = []
    accepted = []
    source_ids = defaultdict(set)

    for row in scan_rows:
        split = row["split"].lower()
        split = "val" if split in {"val", "valid", "validation"} else split
        folder = normalize_folder(row["class_folder"])
        image_path = Path(row["image"]) if row["image"] else None
        label_path = Path(row["label"]) if row["label"] else None

        if image_path is None:
            reports.append({**row, "final_split": "", "result": "ignored_orphan_label"})
            continue

        if label_path is None and folder != "notumor":
            reports.append({**row, "final_split": "", "result": "ignored_missing_label"})
            continue

        try:
            objects = read_yolo_label(label_path) if label_path else []
        except (ValueError, UnicodeError) as error:
            reports.append({**row, "final_split": "", "result": f"ignored_invalid_label: {error}"})
            continue

        source_ids[folder].update(obj[0] for obj in objects)

        if source_id_to_class is None:
            # Mode historique : dossier fiable, un ID source par dossier.
            if folder == "notumor":
                if objects:
                    reports.append({**row, "final_split": "", "result": "ignored_no_tumor_has_boxes"})
                    continue
            elif not objects:
                reports.append({**row, "final_split": "", "result": "ignored_empty_tumor_label"})
                continue
            elif len({obj[0] for obj in objects}) != 1:
                reports.append({**row, "final_split": "", "result": "ignored_multiple_source_classes_in_one_image"})
                continue
            converted_objects = ([(CLASS_IDS[folder], x, y, w, h)
                                  for _, x, y, w, h in objects]
                                 if folder != "notumor" else [])
        else:
            # Mode explicite : l'ID de CHAQUE boîte détermine sa classe.
            unknown_ids = sorted({obj[0] for obj in objects
                                  if obj[0] not in source_id_to_class})
            if unknown_ids:
                reports.append({**row, "final_split": "",
                                "result": f"ignored_unmapped_source_ids_{unknown_ids}"})
                continue

            tumor_objects = []
            has_background_box = False
            for class_id, x, y, w, h in objects:
                class_name = source_id_to_class[class_id]
                if class_name == "notumor":
                    has_background_box = True
                else:
                    tumor_objects.append((CLASS_IDS[class_name], x, y, w, h))

            if folder == "notumor" and tumor_objects:
                reports.append({**row, "final_split": "",
                                "result": "ignored_no_tumor_folder_has_tumor_boxes"})
                continue
            if folder != "notumor" and not tumor_objects:
                reports.append({**row, "final_split": "",
                                "result": "ignored_tumor_folder_without_tumor_boxes"})
                continue
            if (folder != "notumor" and not allow_cross_class_boxes and
                    any(CLASS_NAMES[class_id] != folder for class_id, *_ in tumor_objects)):
                reports.append({**row, "final_split": "",
                                "result": "ignored_source_id_disagrees_with_folder"})
                continue
            if has_background_box and tumor_objects:
                reports.append({**row, "final_split": "",
                                "result": "ignored_mixed_background_and_tumor_boxes"})
                continue
            converted_objects = tumor_objects

        accepted.append({**row, "split": split, "folder": folder, "image_path": image_path,
                         "objects": converted_objects})

    # Sans dictionnaire confirmé, ne pas modifier silencieusement les classes.
    if source_id_to_class is None:
        inconsistent = {folder: sorted(ids) for folder, ids in source_ids.items()
                        if folder != "notumor" and len(ids) > 1}
        if inconsistent:
            raise ValueError(
                f"IDs bruts incohérents : {inconsistent}. "
                "Vérifie data.yaml/classes.txt d'origine, puis passe "
                "source_id_to_class={ID: 'nom_classe', ...} à prepare_dataset()."
            )

    has_test = any(r["split"] == "test" for r in accepted)
    if not has_test:
        by_folder = defaultdict(list)
        for record in accepted:
            if record["split"] == "train":
                by_folder[record["folder"]].append(record)
        rng = random.Random(seed)
        for folder, items in by_folder.items():
            rng.shuffle(items)
            # 1 image par classe reste en train si la classe est trop petite.
            n_test = min(max(1, round(len(items) * test_ratio)), len(items) - 1) if len(items) > 1 else 0
            for record in items[:n_test]:
                record["split"] = "test"

    # Donne priorité à l'ensemble d'évaluation pour éviter les fuites depuis train.
    priority = {"val": 0, "test": 1, "train": 2}
    accepted.sort(key=lambda row: (priority[row["split"]], row["folder"], str(row["image_path"])))
    unique = []
    seen_hashes = set()
    for record in accepted:
        digest = _sha256(record["image_path"])
        if digest in seen_hashes:
            reports.append({"split": record["split"], "class_folder": record["folder"],
                            "image": str(record["image_path"]), "label": record["label"],
                            "final_split": "", "result": "ignored_duplicate_image"})
            continue
        seen_hashes.add(digest)
        unique.append(record)

    if not any(r["split"] == "train" for r in unique):
        raise ValueError("Plus d'images pour Train après contrôle.")
    if not any(r["split"] == "val" for r in unique):
        raise ValueError("Plus d'images pour Val après contrôle.")
    if not any(r["split"] == "test" for r in unique):
        raise ValueError("Pas d'images Test : ajouter un dossier Test ou davantage d'images Train.")
    for split_name in ("train", "val", "test"):
        if not any(r["folder"] != "notumor" for r in unique if r["split"] == split_name):
            raise ValueError(f"Aucune boîte de tumeur dans {split_name} : évaluation impossible.")

    # Nettoie uniquement notre dossier dérivé (jamais data/raw).
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = Counter()
    used_names = set()
    for record in unique:
        split = record["split"]
        folder = record["folder"]
        image_path = record["image_path"]
        # Préfixe de classe pour éviter img1.jpg identique dans deux dossiers.
        prefix = f"{folder}__{image_path.stem}"
        file_name = prefix + image_path.suffix.lower()
        key = (split, prefix.lower())
        if key in used_names:
            # évite même stem avec extension différente dans le même split
            reports.append({"split": split, "class_folder": folder, "image": str(image_path),
                            "label": record["label"], "final_split": "", "result": "ignored_duplicate_filename"})
            continue
        used_names.add(key)

        destination_image = output_root / split / "images" / file_name
        destination_label = output_root / split / "labels" / f"{prefix}.txt"
        shutil.copy2(image_path, destination_image)

        lines = [f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"
                 for class_id, x, y, w, h in record["objects"]]
        destination_label.write_text("\n".join(lines) + ("\n" if lines else ""),
                                     encoding="utf-8")

        counts[(split, folder)] += 1
        reports.append({"split": record["split"], "class_folder": record["folder"],
                        "image": str(image_path), "label": record["label"],
                        "final_split": split, "result": "copied"})

    # Chemins absolus: le notebook marche depuis le dossier projet ou notebooks.
    yaml_path = output_root / "data.yaml"
    yaml_lines = [f'path: "{output_root.as_posix()}"',
                  "train: train/images", "val: val/images", "test: test/images",
                  "nc: 3", "names:", "  0: glioma", "  1: meningioma", "  2: pituitary"]
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    report_path = output_root / "preparation_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class_folder", "image", "label", "final_split", "result"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)

    print("\n===== RÉSUMÉ (images copiées) =====")
    for split in ("train", "val", "test"):
        for folder in ("glioma", "meningioma", "pituitary", "notumor"):
            print(f"{split:5} / {folder:11}: {counts[(split, folder)]}")
    print("\nÉléments ignorés :", sum(not r["result"] == "copied" for r in reports))
    print("Fichier YAML :", yaml_path)
    print("Rapport :", report_path)
    print("IDs bruts observés :", {k: sorted(v) for k, v in source_ids.items()})
    if source_id_to_class is not None:
        print("Correspondance source utilisée (à vérifier dans le data.yaml source) :",
              source_id_to_class)
    return yaml_path, report_path
