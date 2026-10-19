import pandas as pd
from sklearn.model_selection import train_test_split


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