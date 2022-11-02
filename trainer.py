"""
Helper functions
"""
from tensorflow import keras


class Trainer:
    def __init__(self, models, data, batch_size=50, no_epochs=10):
        self.data = data
        self.models = models
        self.batch_size = batch_size
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

    def train_model(self, model_name, all_folds=True):
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
                use_multiprocessing=True,
                epochs=self.no_epochs,
                validation_data=(X_test, y_test),
            )
            histories.append(history)
            if all_folds != True:
                break

        return histories
