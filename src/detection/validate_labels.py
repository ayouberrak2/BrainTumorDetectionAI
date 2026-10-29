"""Lecture et validation de fichiers d'annotations YOLO (simple)."""
from pathlib import Path


def read_yolo_label(label_path):
    """Renvoie les objets [(id, x, y, w, h), ...].

    Les coordonnées sont normalisées entre 0 et 1.
    Un fichier vide est valide pour une image sans tumeur.
    """
    objects = []
    content = Path(label_path).read_text(encoding="utf-8-sig")

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Ligne {line_number}: 5 valeurs attendues")

        try:
            # int('0.0') est refusé volontairement : YOLO attend un entier.
            class_id = int(parts[0])
            x, y, w, h = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(f"Ligne {line_number}: valeur non numérique") from error

        if class_id < 0:
            raise ValueError(f"Ligne {line_number}: identifiant de classe négatif")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Ligne {line_number}: coordonnées hors de [0, 1]")
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
            raise ValueError(f"Ligne {line_number}: boîte hors des limites horizontales")
        if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            raise ValueError(f"Ligne {line_number}: boîte hors des limites verticales")
        objects.append((class_id, x, y, w, h))

    return objects
