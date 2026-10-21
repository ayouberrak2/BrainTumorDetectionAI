
import tensorflow as tf

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32



def load_classification_image(
        data_dir,
        image_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE
):
    root = data_dir

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{root}/train",
        image_size= image_size,
        batch_size = batch_size,
        shuffle = True,
        seed=42,
        label_mode = "int",
    )

    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
            f"{root}/valid",
            image_size= image_size,
            batch_size = batch_size,
            shuffle = False,
            class_names=class_names,
            label_mode = "int",
        )


    test_ds = tf.keras.utils.image_dataset_from_directory(
            f"{root}/test",
            image_size= image_size,
            batch_size = batch_size,
            shuffle = False,
            class_names=class_names,
            label_mode = "int",
        )


    return train_ds, val_ds,test_ds,class_names

