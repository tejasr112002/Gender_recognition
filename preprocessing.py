import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random

genders = ["m", "f"]
ages = [
    "(0, 2)",
    "(4, 6)",
    "(8, 12)",
    "(15, 20)",
    "(25, 32)",
    "(38, 43)",
    "(48, 53)",
    "(60, 100)",
]


# read txt file with pandas
def get_dataframe(path, frontal=False):
    if frontal:
        dfs = [
            pd.read_csv(f"{path}/fold_frontal_{i}_data.txt", sep="\t") for i in range(5)
        ]
    else:
        dfs = [pd.read_csv(f"{path}/fold_{i}_data.txt", sep="\t") for i in range(5)]
    df = pd.concat(dfs, ignore_index=True)

    # selct only the columns we need
    image_path = (
        df["user_id"].astype(str)
        + "/"
        + "landmark_aligned_face."
        + df["face_id"].astype(str)
        + "."
        + df["original_image"].astype(str)
    )
    df.insert(0, "image_path", image_path)
    # df = df[['image_path','age','gender']]

    # drop rows with missing values
    df = df.dropna()

    # drop rows with invalid genders
    df = df[[x in ["m", "f"] for x in df.gender]]

    # drop rows with invalid ages
    df = df[[x in ages for x in df.age]]

    # convert age to the index in ages
    df.age = df.age.apply(lambda x: ages.index(x))
    # convert gender to the index in genders
    df.gender = df.gender.apply(lambda x: genders.index(x))

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def barplots_of_labels(df):
    # make age and gender barplots in subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    df.age.value_counts().plot(kind="bar", ax=ax[0, 0])
    df.gender.value_counts().plot(kind="bar", ax=ax[0, 1])
    df.age[df.gender == 1].value_counts().plot(kind="bar", ax=ax[1, 0])
    df.age[df.gender == 0].value_counts().plot(kind="bar", ax=ax[1, 1])


def create_dataset(df, n_max=False, new_size=False):
    # create a dataset from a dataframe
    dataset = []
    N = n_max if n_max else len(df)
    new_size = new_size if new_size else (224, 224)
    for i in range(N):
        # read image
        image = cv2.imread(f"data/aligned/{df.image_path[i]}")
        # resize image
        image = cv2.resize(image, new_size)
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # extract gender and age from df and append tuple
        dataset.append((image, df.gender[i], df.age[i]))
    return dataset


# function to split dataset
def split_dataset(dataset, test_size=0.3):
    # split dataset into train and test
    random.shuffle(dataset)
    N = len(dataset)
    N_test = int(N * test_size)
    N_train = N - N_test
    train_dataset = dataset[:N_train]
    test_dataset = dataset[N_train:]
    # split train and test into X and y
    X_train, gender_train, age_train = zip(*train_dataset)
    X_test, gender_test, age_test = zip(*test_dataset)
    # convert to numpy arrays
    X_train = np.array(X_train)
    gender_train = np.array(gender_train)
    age_train = np.array(age_train)
    X_test = np.array(X_test)
    gender_test = np.array(gender_test)
    age_test = np.array(age_test)
    return X_train, gender_train, age_train, X_test, gender_test, age_test


def get_preprocessed_dataset(
    path="data", frontal=False, n_max=False, new_size=False, test_size=0.3
):
    df = get_dataframe("data", frontal=frontal)
    ds = create_dataset(df, n_max=n_max, new_size=new_size)
    return split_dataset(ds, test_size=test_size)


if __name__ == "__main__":
    (
        X_train,
        gender_train,
        age_train,
        X_test,
        gender_test,
        age_test,
    ) = get_preprocessed_dataset(
        path="data", frontal=False, n_max=25, new_size=(256, 256), test_size=0.3
    )
