import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
import warnings
import pickle
import glob

from preprocessing import DataPreprocessing
from models import baseline_model, main_model, CNN_transfer, CNN_multitask
from trainer import Trainer
import plot_functions as pf

DATA_PATH = "./Data/"
CHECKPOINT_PATH = "./Models/"
HISTORIES_PATH = "./Histories/"


# turn off warnings
def turn_off_warnings():
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")


def set_seed(SEED):
    """Set seed for reproducibility."""
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_training_results(model_name):
    model = keras.models.load_model(CHECKPOINT_PATH + f"{model_name}/")
    histories = []
    for path in sorted(glob.glob(f"./{HISTORIES_PATH }/{model_name}/*")):
        with open(path, "rb") as f:
            histories.append(pickle.load(f))
    return model, histories
