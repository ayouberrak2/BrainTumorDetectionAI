"""Entraîner YOLOv8 avec une configuration lisible dans un YAML."""
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


def train_yolov8(data_yaml, config_path, project_dir):
    """Lance l'entraînement et renvoie le chemin de best.pt."""
    with open(config_path, encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    model_name = settings.pop("model")
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Modèle préentraîné :", model_name)
    print("Appareil :", device)

    model = YOLO(model_name)
    model.train(
        data=str(Path(data_yaml).resolve()),
        device=device,
        project=str(Path(project_dir).resolve()),
        name="yolov8_brain_tumor",
        exist_ok=True,
        **settings,
    )

    best_path = Path(project_dir).resolve() / "yolov8_brain_tumor" / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Le fichier best.pt est introuvable : {best_path}")
    print("Meilleur modèle :", best_path)
    return best_path
