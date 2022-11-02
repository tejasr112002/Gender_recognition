import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import warnings

# turn off warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


def set_seed(SEED):
    """Set seed for reproducibility."""
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)