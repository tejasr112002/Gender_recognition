from initialsation import *
from preprocessing import DataPreprocessing
from models import main_model, baseline_model, CNN_classic, CNN_transfer, CNN_multitask
from trainer import Trainer
import plot_functions as pf

# set seed for reproducibility
set_seed(42)


IMG_SIZE = 256

preprocessing = DataPreprocessing(path="data", n_max=25, new_size=(IMG_SIZE, IMG_SIZE))
datasets = (
    preprocessing.get_cv_datasets()
)  # list of (X_train, (y_train_gender, y_train_age), X_test, (y_train_gender, y_train_age))

models = {
    "Main Model": (main_model(IMG_SIZE), "gender", None),
    # "CNN classic": (CNN_classic(img_size),"gender", None),
    # "CNN transfer": (CNN_transfer(img_size),"gender", None),
    # "CNN multitask": (CNN_multitask(img_size),"multi" 0.5),
}

trainer = Trainer(models, datasets, no_epochs=10)
histories = trainer.train_model("Main Model", all_folds=False)

pf.plot_history(histories[0], "Main Model")
