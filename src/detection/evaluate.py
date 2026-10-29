"""Mesurer précision, rappel, mAP50 et mAP50-95 sur validation ou test."""
from pathlib import Path
import json

import torch
from ultralytics import YOLO


def evaluate_yolov8(model_path, data_yaml, project_dir, split="test", imgsz=512, batch=8):
    """Évalue best.pt et sauvegarde un rapport JSON + graphiques YOLO."""
    if split not in {"val", "test"}:
        raise ValueError("split doit être 'val' ou 'test'")

    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.val(
        data=str(Path(data_yaml).resolve()),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(Path(project_dir).resolve()),
        name=f"evaluation_{split}",
        exist_ok=True,
        plots=True,
        workers=0,
    )

    metrics = {
        "split": split,
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
    }
    out_dir = Path(project_dir).resolve() / f"evaluation_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Graphiques et rapport :", out_dir)
    return metrics
