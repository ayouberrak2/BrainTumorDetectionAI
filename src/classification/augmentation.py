import tensorflow as tf

def build_data_augmantation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.GaussianNoise(0.02),
    ],name = "data_augmentation")



