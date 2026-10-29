from pathlib import Path

import numpy as np
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_true_list = []
    y_pred_proba_list = []

    # 1. Récupérer labels et prédictions ensemble
    for images, labels in dataset:

        probabilities = model(
            images,
            training=False
        ).numpy()

        y_true_list.append(
            labels.numpy().reshape(-1)
        )

        y_pred_proba_list.append(
            probabilities
        )

    # 2. Fusionner les résultats
    y_true = np.concatenate(
        y_true_list
    ).astype(int)

    y_pred_proba = np.concatenate(
        y_pred_proba_list
    )

    # 3. Classes prédites
    y_pred = np.argmax(
        y_pred_proba,
        axis=1
    )

    # 4. Calculer les métriques
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4
    )


    # 7. Matrice de confusion
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"Matrice de confusion - {dataset_name}"
    )

    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    confusion_path = output_path / (
        f"confusion_matrix_{dataset_name.lower()}.png"
    )

    fig.savefig(
        confusion_path,
        dpi=150,
        bbox_inches="tight"
    )

    # plt.show()
    plt.close(fig)

    report_path = output_path / (
        f"classification_report_{dataset_name.lower()}.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    # 9. Résultats
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
    return results