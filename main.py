from initialsation import *
from preprocessing import DataPreprocessing
from models import main_model, baseline_model, CNN_classic, CNN_transfer, CNN_multitask

# set seed for reproducibility
set_seed(42)


IMG_SIZE = 256

preprocessing = DataPreprocessing(path="data", n_max=25, new_size=(IMG_SIZE, IMG_SIZE))
datasets = preprocessing.get_cv_datasets()

models = {
    "Main Model": (main_model(IMG_SIZE), None),
    # "CNN classic": (CNN_classic(img_size), None),
    # "CNN transfer": (CNN_transfer(img_size), None),
    # "CNN multitask": (CNN_multitask(img_size), 0.5),
}

model = models["Main Model"][0]
model.summary()
