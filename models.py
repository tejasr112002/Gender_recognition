import tensorflow as tf
from tensorflow import keras
from keras import layers

# MODELS ------------------------------------------------


def model_gender_classification(img_size, reducer=1, seed=42):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    inputs_crop = _add_data_augmenting(inputs, 227, seed=seed)
    main_branch = _get_main_branch(inputs_crop, reducer=reducer)
    gender_branch = _add_gender_branch(main_branch)
    model = tf.keras.Model(inputs=inputs, outputs=gender_branch)
    return model


def model_age_classification(img_size, reducer=1, seed=42):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    inputs_crop = _add_data_augmenting(inputs, 227, seed=seed)
    main_branch = _get_main_branch(inputs_crop, reducer=reducer)
    age_branch = _add_age_branch(main_branch)
    model = tf.keras.Model(inputs=inputs, outputs=age_branch)
    return model


def model_multitask_classification(img_size, reducer=1, seed=42):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    inputs_crop = _add_data_augmenting(inputs, 227, seed=seed)
    main_branch = _get_main_branch(inputs_crop, reducer=reducer)
    gender_branch = _add_gender_branch(main_branch)
    age_branch = _add_age_branch(main_branch)
    model = tf.keras.Model(inputs=inputs, outputs=[gender_branch, age_branch])
    return model


def model_transfer_multitask_classification(img_size, seed=42):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    inputs_crop = _add_data_augmenting(inputs, 227, seed=seed)
    main_branch = _get_imagenet_base_model(inputs_crop, 227)
    gender_branch = _add_gender_branch(main_branch)
    age_branch = _add_age_branch(main_branch)
    model = tf.keras.Model(inputs=inputs, outputs=[gender_branch, age_branch])
    return model


# HELP FUNCTIONS (not loaded by *) ------------------------------------------------


def _get_imagenet_base_model(inputs, img_size):

    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(img_size, img_size, 3),
        include_top=False,
    )
    # freeze all layers except the last 10
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    return layers.Flatten()(base_model(inputs))


def _add_data_augmenting(inputs, crop_size, seed=42):
    inputs = layers.RandomCrop(crop_size, crop_size, seed=seed)(inputs)
    inputs = layers.Rescaling(scale=1 / 127.5, offset=-1)(inputs)
    return inputs


def _get_main_branch(inputs, reducer=1):
    model = keras.Sequential(
        [
            layers.Conv2D(96 // reducer, 7, activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Conv2D(256 // reducer, 5, activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Conv2D(384 // reducer, 3, activation="relu"),
            layers.MaxPooling2D((3, 3), strides=2),
            layers.Lambda(tf.nn.local_response_normalization),
        ]
    )
    return layers.Flatten()(model(inputs))


def _add_gender_branch(main_branch):
    gender_branch = layers.Dense(128, activation="relu")(main_branch)
    gender_branch = layers.Dropout(0.5)(gender_branch)
    gender_branch = layers.Dense(128, activation="relu")(gender_branch)
    gender_branch = layers.Dropout(0.5)(gender_branch)
    output = layers.Dense(1, activation="sigmoid", name="gender")(gender_branch)
    return output


def _add_age_branch(main_branch):
    age_branch = layers.Dense(128, activation="relu")(main_branch)
    age_branch = layers.Dropout(0.5)(age_branch)
    age_branch = layers.Dense(128, activation="relu")(age_branch)
    age_branch = layers.Dropout(0.5)(age_branch)
    output = layers.Dense(8, activation="softmax", name="age")(age_branch)
    return output

