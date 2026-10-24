from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def evaluate_googlenet(
    model,
    dataset,
    class_names: list[str],
    dataset_name: str = "TEST",
    output_dir: str = "artifacts/evaluation",
):
    """
    Évalue un modèle GoogLeNet de classification d'images.

    Args:
        model: Modèle entraîné.
        dataset: Dataset TensorFlow à évaluer.
        class_names: Noms des classes dans l'ordre des labels.
        dataset_name: Nom du dataset évalué.
        output_dir: Répertoire de sauvegarde des résultats.

    Returns:
        results: Dictionnaire contenant les métriques.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Récupérer les vrais labels
    y_true = np.concatenate([
        labels.numpy().reshape(-1)
        for _, labels in dataset
    ]).astype(int)

    # 2. Effectuer les prédictions
    y_pred_proba = model.predict(dataset, verbose=1)

    # 3. Convertir les probabilités en classes prédites
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 4. Calculer les métriques
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    recall_weighted = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    f1_weighted = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 5. Afficher les résultats globaux
    print(f"\n{'=' * 55}")
    print(f"ÉVALUATION DU MODÈLE GOOGLENET : {dataset_name}")
    print(f"{'=' * 55}")

    print(f"Nombre d'images : {len(y_true)}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision macro : {precision_macro:.4f}")
    print(f"Recall macro    : {recall_macro:.4f}")
    print(f"F1-score macro  : {f1_macro:.4f}")

    print("\nMétriques pondérées :")
    print(f"Precision       : {precision_weighted:.4f}")
    print(f"Recall          : {recall_weighted:.4f}")
    print(f"F1-score        : {f1_weighted:.4f}")

    # 6. Rapport détaillé par classe
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    print("\nRapport de classification :")
    print(report)

    # 7. Matrice de confusion
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(f"Matrice de confusion - {dataset_name}")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    confusion_path = output_path / (
        f"confusion_matrix_{dataset_name.lower()}.png"
    )

    fig.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # 8. Sauvegarder le rapport de classification
    report_path = output_path / (
        f"classification_report_{dataset_name.lower()}.txt"
    )

    report_path.write_text(report, encoding="utf-8")

    # 9. Retourner les métriques
    results = {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }

    print(f"\nMatrice sauvegardée dans : {confusion_path}")
    print(f"Rapport sauvegardé dans : {report_path}")

    return results