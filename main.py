from initialsation import *

SEED = 42
IMG_SIZE = 256

# set seed for reproducibility and
set_seed(SEED)
turn_off_warnings()

preprocessing = DataPreprocessing(DATA_PATH, n_max=5, new_size=(IMG_SIZE, IMG_SIZE))
# list of (X_train, (y_train_gender, y_train_age), X_test, (y_train_gender, y_train_age))
datasets = preprocessing.get_cv_datasets()

models = {  # dictionary of models - name : (model, target, gamma)
    "Baseline": (baseline_model(IMG_SIZE), "gender", None),
    "Main Model": (main_model(IMG_SIZE, reducer=1, seed=42), "gender", None),
    "CNN transfer": (CNN_transfer(IMG_SIZE), "gender", None),
    "CNN multitask": (CNN_multitask(IMG_SIZE), "multi", 0.5),
}

trainer = Trainer(
    models,
    datasets,
    checkpoint_filepath=CHECKPOINT_PATH,
    histories_filepath=HISTORIES_PATH,
    no_epochs=10,
    batch_size=50,
    patience=10,
    all_folds=False,
    use_multiprocessing=False,
)

# train
# for model in models.keys():
#     trainer.train_model(model)

trainer.train_model("CNN multitask")
