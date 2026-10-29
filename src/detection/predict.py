"""Tester une image avec le meilleur modèle entraîné."""
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO


def predict_image(model_path, image_path, output_dir, confidence=0.25):
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        save=True,
        project=str(Path(output_dir).resolve()),
        name="predictions",
        exist_ok=True,
    )
    # results[0].plot() renvoie une image en BGR -> RGB pour matplotlib.
    image_with_boxes = results[0].plot()[..., ::-1]
    plt.figure(figsize=(7, 7))
    plt.imshow(image_with_boxes)
    plt.axis("off")
    plt.title("Prédictions YOLOv8")
    plt.show()
    print("Images sauvegardées dans :", Path(output_dir) / "predictions")
    return results
