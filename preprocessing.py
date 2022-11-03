"""
Dataset preprocessing class

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split, KFold


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


# make Dataset preprocessing class
class DataPreprocessing:
    def __init__(self, path, frontal=False, n_max=None, new_size=None):
        self.genders = ["m", "f"]
        self.ages = [
            "(0, 2)",
            "(4, 6)",
            "(8, 12)",
            "(15, 20)",
            "(25, 32)",
            "(38, 43)",
            "(48, 53)",
            "(60, 100)",
        ]
        self.new_size = new_size
        self.n_max = n_max
        self.frontal = frontal
        self.path = path

    def _get_preprocessed_dataframes(self):
        dfs = self._load_raw_data()
        dfs_preprocesed = [self._preprocess_dataframe(df).sample(frac=1) for df in dfs]
        # sample n_max rows
        if self.n_max is not None:
            dfs_preprocesed = [df.sample(n=self.n_max) for df in dfs_preprocesed]
        return dfs_preprocesed

    def _load_raw_data(self):
        if self.frontal:
            dfs = [
                pd.read_csv(f"{self.path}fold_frontal_{i}_data.txt", sep="\t")
                for i in range(5)
            ]
        else:
            dfs = [
                pd.read_csv(f"{self.path}fold_{i}_data.txt", sep="\t") for i in range(5)
            ]
        # df = pd.concat(dfs, ignore_index=True)
        return dfs

    def _preprocess_dataframe(self, df):
        # add image paths
        image_path = self._get_image_paths(df)
        df.insert(0, "image_path", image_path)
        # select only the columns we need
        df = df[["image_path", "age", "gender"]]

        # drop rows with missing values
        df = df.dropna()

        # drop rows with invalid genders
        df = df[[x in self.genders for x in df.gender]]

        # drop rows with invalid ages
        df = df[[x in self.ages for x in df.age]]

        # convert age and gender labels to indexes
        df.age = df.age.apply(lambda x: ages.index(x))
        df.gender = df.gender.apply(lambda x: genders.index(x))

        return df

    def _get_image_paths(self, df):
        image_path = (
            df["user_id"].astype(str)
            + "/landmark_aligned_face."
            + df["face_id"].astype(str)
            + "."
            + df["original_image"].astype(str)
        )
        return image_path

    def _dataframe_to_dataset(self, df):
        dataset = []
        for row in df.itertuples():
            # read image
            image = cv2.imread(f"data/aligned/{row.image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.new_size is not None:
                image = cv2.resize(image, self.new_size)
            # extract gender and age from df and append tuple
            dataset.append((image, row.gender, row.age))
        # convert to data set to a tuple of numpy arrays
        X, y1, y2 = tuple(map(np.array, zip(*dataset)))
        return (X, (y1, y2))

    def _map_to_datasets(self, dfs):
        return sum(tuple(map(self._dataframe_to_dataset, dfs)), ())  # flatten

    def get_age_label(self, age):
        return self.ages[age]

    def get_gender_label(self, gender):
        return self.genders[gender]

    def get_classes(self):
        return genders, ages

    def get_cv_splits(self):
        # returns list of (X_train, (y_train_gender, y_train_age), X_test, (y_train_gender, y_train_age))
        dfs = self._get_preprocessed_dataframes()
        cv_splits = []
        for i in range(5):
            dfs_copy = dfs[:]
            test_df = dfs_copy.pop(i)
            train_df = pd.concat(dfs_copy, ignore_index=True)
            cv_splits.append(self._map_to_datasets([train_df, test_df]))

        return cv_splits

    # def get_train_test_split(self, test_size=0.3, random_state=42):
    #     train_df, test_df = train_test_split(
    #         self.df, test_size=test_size, random_state=random_state
    #     )
    #     return self._map_to_datasets([train_df, test_df])

    # def _get_cv_splits(self, n_splits=5, random_state=42):
    #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    #     # make a list of (train_df, test_df) tuples
    #     cv_splits = [
    #         self._map_to_datasets([self.df.iloc[train_index], self.df.iloc[test_index]])
    #         for train_index, test_index in kf.split(self.df)
    #     ]
    #     return cv_splits


if __name__ == "__main__":
    Data = DatasetPreprocessing(
        path="data", frontal=True, n_max=25, new_size=(256, 256)
    )
    # get train and test datasets [(X_train, y_train_gender, y_train_age), (X_test, y_test, y_train_gender, y_train_age)]
    # train_test_ds = Data.get_train_test_datasets(test_size=0.3, random_state=42) # use for 70/30 split
    train_test_ds = Data.get_cv_splits(
        n_splits=5, random_state=42
    )  # use for 5-fold cross validation
