#%%
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import pandas as pd
import os
from plot_functions import plot_histories

DATA_PATH = "./Data/"
CHECKPOINT_PATH = "./Models/"
HISTORIES_PATH = "./Histories/"

# make figure folder if it doesn't exist
if not os.path.exists("./Figures/"):
    os.makedirs("./Figures/")


def load_histories(model_name):
    histories = []
    for path in sorted(glob.glob(f"{HISTORIES_PATH}/{model_name}/*")):
        with open(path, "rb") as f:
            histories.append(pickle.load(f))
    return histories


def load_model(model_name):
    return keras.models.load_model(CHECKPOINT_PATH + model_name)


def gen_acc(histories, metric="val_accuracy"):
    """Calculate generalization error for each fold."""
    gen_error = []
    for history in histories:
        gen_error.append(np.max(history[metric]))
    return np.array([gen_error]).mean() * 100, np.array([gen_error]).std() * 100


#%%
results = pd.DataFrame(
    columns=[
        "model",
        "gender_gen_acc",
        "gender_gen_acc_std",
        "age_gen_acc",
        "age_gen_acc_std",
    ]
)

for model_name in ["GenderR4", "AgeR4", "Multitask", "Transfer"]:
    histories = load_histories(model_name)
    if model_name in ["Transfer", "Multitask"]:
        m_gender, s_gender = gen_acc(histories, metric="val_age_accuracy")
        m_age, s_age = gen_acc(histories, metric="val_age_accuracy")
    elif model_name == "AgeR4":
        m_gender, s_gender = [np.nan, np.nan]
        m_age, s_age = gen_acc(histories, metric="val_accuracy")
    elif model_name == "GenderR4":
        m_gender, s_gender = gen_acc(histories, metric="val_accuracy")
        m_age, s_age = [np.nan, np.nan]

    print(model_name)

    results = results.append(
        {
            "model": model_name,
            "gender_gen_acc": round(m_gender, 1),
            "gender_gen_acc_std": round(s_gender, 1),
            "age_gen_acc": round(m_age, 1),
            "age_gen_acc_std": round(s_age, 1),
        },
        ignore_index=True,
    )

    plot_histories(histories)
    plt.savefig(f"./Figures/{model_name}.pdf", bbox_inches="tight")

results.to_csv("./Figures/results.csv", index=False)

# %%
