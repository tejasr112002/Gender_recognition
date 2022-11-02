import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np


def main_model(img_size, seed=42):
    model = keras.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            # add data augmentation here
            layers.RandomCrop(227, 227, seed=seed),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(96, 7, padding="same", activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Conv2D(256, 5, padding="same", activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Conv2D(384, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="CNN_classic",
    )
    return model


def main_model(img_size, seed=42):
    model = keras.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            # add data augmentation here
            layers.RandomCrop(227, 227, seed=seed),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(43, 7, padding="same", activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Conv2D(128, 5, padding="same", activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Conv2D(192, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="CNN_classic",
    )
    return model


# make baseline model that just predict the majority class
def baseline_model(img_size):
    model = keras.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="baseline",
    )
    return model


# function to create standard CNN network
# idea: expand function such that its easy to change the architechture
def CNN_classic(img_size):
    model = keras.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            layers.Rescaling(1.0 / 255),
            # layers.Rescaling(scale = 1./127.5, offset = -1), do this if we want to be consistent with the transfer learning network
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="CNN_classic",
    )
    return model


def CNN_transfer(img_size):
    # follows this guide https://keras.io/guides/transfer_learning/

    # initialize base model from keras API
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(img_size, img_size, 3),
        include_top=False,
    )

    # freeze weights
    base_model.trainable = False

    # define rest of model from here
    input = layers.Input(shape=(img_size, img_size, 3))
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(input)
    x = base_model(
        x
    )  # ,training = false) maybe neccesary if we want to fine tune model
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(input, outputs, name="CNN_transfer")
    return model


# function to create the architechture of the multitask network
def CNN_multitask(img_size):

    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3), name="input")
    main_branch = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(inputs)
    main_branch = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(
        main_branch
    )
    main_branch = tf.keras.layers.MaxPooling2D()(main_branch)
    main_branch = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(
        main_branch
    )
    main_branch = tf.keras.layers.MaxPooling2D()(main_branch)
    main_branch = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(
        main_branch
    )
    main_branch = tf.keras.layers.Flatten()(main_branch)
    main_branch = tf.keras.layers.Dense(128, activation="relu")(main_branch)

    task_1_branch = tf.keras.layers.Dense(256, activation="relu")(main_branch)
    task_1_branch = tf.keras.layers.Dense(128, activation="relu")(task_1_branch)
    task_1_branch = tf.keras.layers.Dense(1, activation="sigmoid", name="gender")(
        task_1_branch
    )

    task_2_branch = tf.keras.layers.Dense(256, activation="relu")(main_branch)
    task_2_branch = tf.keras.layers.Dense(128, activation="relu")(task_2_branch)
    task_2_branch = tf.keras.layers.Dense(8, activation="softmax", name="age")(
        task_2_branch
    )

    model = tf.keras.Model(
        inputs=inputs, outputs=[task_1_branch, task_2_branch], name="CNN_multitask"
    )
    return model


if __name__ == "__main__":
    model = main_model(227)
    model.summary()
