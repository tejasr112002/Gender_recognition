import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

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
    genders = ["m", "f"]
    df = df[[x in ["m", "f"] for x in df.gender]]

    # drop rows with invalid ages
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
    df = df[[x in ages for x in df.age]]

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def barplots_of_labels(df):
    # make age and gender barplots in subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    df.age.value_counts().plot(kind='bar', ax=ax[0,0])
    df.gender.value_counts().plot(kind='bar', ax=ax[0,1])
    df.age[df.gender == 'f'].value_counts().plot(kind='bar', ax=ax[1,0])
    df.age[df.gender == 'm'].value_counts().plot(kind='bar', ax=ax[1,1])


if __name__ == "__main__":
    df = get_dataframe("data", frontal=False)
