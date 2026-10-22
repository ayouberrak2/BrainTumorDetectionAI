
import tensorflow_hub as hub
import tf_keras as keras
from tf_keras import layers
import kagglehub


def build_googlenet(
    num_classes: int,
    input_shape: tuple = (224, 224, 3),
    trainable_base: bool = False,
    learning_rate: float = 1e-3,
):
    if num_classes < 2:
        raise ValueError("num_classes doit être supérieur ou égal à 2.")

    # Télécharger le modèle GoogLeNet préentraîné
    model_path = kagglehub.model_download(
        "google/inception-v1/tensorFlow2/feature-vector/1"
    )

    # Backbone préentraîné
    base_model = hub.KerasLayer(
        model_path,
        trainable=trainable_base,
        name="googlenet_feature_extractor",
    )

    # Entrée de l'image
    inputs = keras.Input(shape=input_shape, name="image")

    # Extraction des caractéristiques
    x = base_model(inputs)

    # Tête de classification
    x = layers.Dense(
        256,
        activation="relu",
        name="classifier_dense",
    )(x)

    x = layers.Dropout(
        0.4,
        name="classifier_dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="brainscan_googlenet",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
