
from pathlib import Path
import tf_keras as keras


def train_googlenet(
    model,
    train_ds,
    valid_ds,
    epochs: int = 30,
    output_dir: str = "artifacts/models",
    patience: int = 5,
):
    """
    Entraîne le modèle GoogLeNet préentraîné.

    Args:
        model: Modèle créé avec build_googlenet().
        train_ds: Dataset d'entraînement.
        valid_ds: Dataset de validation.
        epochs: Nombre maximal d'époques.
        output_dir: Répertoire de sauvegarde.
        patience: Nombre d'époques sans amélioration avant l'arrêt.

    Returns:
        model: Modèle entraîné.
        history: Historique de l'entraînement.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / "googlenet_best.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Meilleur modèle sauvegardé dans : {checkpoint_path}")

    return model, history
