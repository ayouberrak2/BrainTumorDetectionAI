import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

def split_dataset(manifest,):
    if manifest.empty:
        raise ValueError("Le manifeste est vide.")


    train_df, temporary_df = train_test_split(
        manifest,
        test_size=0.30,
        random_state=42,
        stratify=manifest["label"],
    )

    val_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=42,
        stratify=temporary_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


from pathlib import Path
import shutil


def export_classification_splits(splits,output_dir):
    root = Path(output_dir)

    for split_name, df in splits.items():

        if split_name not in {"train", "valid", "test"}:
            raise ValueError(f"Split inconnu : {split_name}")

        for row in df.itertuples(index=False):
            source = Path(row.path)

            destination = (
                root / split_name / row.label / source.name
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if destination.exists():
                raise FileExistsError(
                    f"Destination déjà existante : {destination}"
                )

            shutil.copy2(source, destination)

