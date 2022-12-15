from loguru import logger
from collections import defaultdict
import pandas as pd
from .base import BaseAutoML
import numpy as np
import autosklearn.regression as AutoSklearn

from sklearn.metrics import (mean_squared_error,
                             mean_absolute_percentage_error,
                             mean_absolute_error)


class AutoSklearnAutoML(BaseAutoML):
    def __init__(self, configs):
        super(AutoSklearnAutoML, self).__init__(
            configs["num_iteration"], configs["train_size"])

        automl_config = configs["automl"]
        self.num_fold = configs["num_cross_validation_fold"]

    def fit(self, X_train, y_train):
        logger.info(
            f"Start fit by auto-sklearn on training set with kfold {self.num_fold}")
        self.pipeline_optimizer.fit(X_train, y_train)
        i = np.argmax(self.pipeline_optimizer.cv_results_['mean_test_score'])
        logger.info(f"Finish training")
        logger.info(
            f"Best pipeline resulst {self.pipeline_optimizer.cv_results_}")
        # return best pipeline
        return self.pipeline_optimizer.cv_results_['params'][i]['regressor:__choice__']

    def infer(self, X_test):
        logger.info("Start inference on test data")
        y_pred = self.pipeline_optimizer.predict(X_test)
        return y_pred

    def eval(self, y_test, y_pred):
        """Calculate metrics RMSE, MAE, MPE, MAPE"""
        logger.info("Start evaluation")
        rmse_loss = mean_squared_error(
            y_test, y_pred, squared=False)  # squared = False -> rmse
        mae_loss = mean_absolute_error(y_test, y_pred)
        mape_loss = mean_absolute_percentage_error(y_test, y_pred)

        return {"RMSE": rmse_loss,
                "MAE": mae_loss,
                "MAPE": mape_loss}

    def preprocess_data(self, data):
        return data  # dont need to preprocess so just return the original\

    def save_record(self, recorder, save_record):
        df = defaultdict(list)
        for iter, info in recorder.items():
            df["iterations"].append(iter)
            df["best_pipeline"].append(info[0])
            df["test_scores"].append(info[1])

        df = pd.DataFrame(df)
        df.to_csv(save_record, index=False)

    def run(self, dataset_path, save_record=None):
        # init recorer for saving results
        recorder = {}

        # Start
        X, y = self.setup_data(dataset_path)
        X, y = self.preprocess_data((X, y))
        for iter in range(self.n):
            self.pipeline_optimizer = AutoSklearn.AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                delete_tmp_folder_after_terminate=True,
                n_jobs=-1,
                disable_evaluator_output=False,
                resampling_strategy="cv",
                resampling_strategy_arguments={
                    "train_size": self.train_size,
                    "folds": self.num_fold
                },
                seed=iter
            )
            logger.info(f"Run iteration {iter}")
            # split train/test dataset
            X_train, X_test, y_train, y_test = self.split_dataset(X, y)

            # training automl model, ausklearn already have kfold
            best_pipeline = self.fit(X_train, y_train)

            # run infer on testset
            y_pred = self.infer(X_test)

            # run evaluation
            metric_scores = self.eval(y_test, y_pred)
            logger.info(f"{metric_scores}")

            # save results
            recorder[iter] = [best_pipeline, metric_scores]

        if (save_record):
            self.save_record(recorder, save_record)
