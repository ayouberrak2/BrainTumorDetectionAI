import tensorflow as tf

from src.classification.augmentation import build_data_augmantation

augmentation = build_data_augmantation()


def prepare_train(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = augmentation(images, training=True)
    return images, labels


def prepare_evaluation(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels