"""
Helper functions
"""
from tensorflow import keras
import pickle
import os


class Trainer:
    def __init__(
        self,
        models,
        data,
        checkpoint_filepath="./Models/",
        histories_filepath="./Histories/",
        batch_size=50,
        no_epochs=10,
        patience=10,
        all_folds=True,
        use_multiprocessing=True,
    ):
        self.data = data
        self.models = models
        self.patience = patience
        self.batch_size = batch_size
        self.all_folds = all_folds
        self.use_multiprocessing = use_multiprocessing
        self.checkpoint_filepath = checkpoint_filepath
        self.histories_filepath = histories_filepath
        self.no_epochs = no_epochs
        self.optimizer = "adam"
        self.metrics = ["accuracy"]

    def compile_model(self, model_name):
        model = self.models[model_name][0]  # get model
        target = self.models[model_name][1]  # get target
        gamma = self.models[model_name][2]  # get gamma
        if target == "multi":
            loss = {
                "gender": keras.losses.BinaryCrossentropy(),
                "age": keras.losses.SparseCategoricalCrossentropy(),
            }
            loss_weights = {"gender": gamma, "age": 1 - gamma}
            model.compile(
                loss=loss,
                loss_weights=loss_weights,
                optimizer=self.optimizer,
                metrics=self.metrics,
            )
        elif target == "gender":
            model.compile(
                loss=keras.losses.BinaryCrossentropy(),
                optimizer=self.optimizer,
                metrics=self.metrics,
            )
        elif target == "age":
            model.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer=self.optimizer,
                metrics=self.metrics,
            )

    def _callbacks(self, checkpoint_filepath, target):
        if target == "multi":
            monitor = "val_loss"
            mode = 'min'
        else:
            monitor = "val_accuracy"
            mode = 'max'
        return [
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=self.patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor=monitor,
                mode=mode,
                save_best_only=True,
            ),
        ]

    def _save_histories(self, histories, model_name):
        dir = self.histories_filepath + model_name
        if not os.path.exists(dir):
            os.makedirs(dir)
        for i, history in enumerate(histories):
            path = dir + f"/fold_{i+1}"
            with open(path, "wb") as f:
                pickle.dump(history.history, f)

    def train_model(self, model_name):
        self.compile_model(model_name)
        model = self.models[model_name][0]  # get model
        target = self.models[model_name][1]  # get target
        histories = []
        for data_split in self.data:
            X_train, y_train, X_test, y_test = data_split
            if target == "gender":
                y_train = y_train[0]
                y_test = y_test[0]
            elif target == "age":
                y_train = y_train[1]
                y_test = y_test[1]
            history = model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                use_multiprocessing=self.use_multiprocessing,
                epochs=self.no_epochs,
                validation_data=(X_test, y_test),
                callbacks=self._callbacks(
                    self.checkpoint_filepath + model_name + "/", target
                ),
            )
            histories.append(history)
            if self.all_folds != True:
                break

        self._save_histories(histories, model_name)
