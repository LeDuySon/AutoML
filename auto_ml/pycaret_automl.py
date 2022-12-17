from loguru import logger
from collections import defaultdict
import pandas as pd
from .base import BaseAutoML
import pycaret.regression as pycaret
from sklearn.metrics import (mean_squared_error, mean_absolute_error)
from utils.mape import mean_absolute_percentage_error

class PycaretAutoML(BaseAutoML):
    def __init__(self, configs):
        super(PycaretAutoML, self).__init__(configs["num_iteration"], configs["train_size"])
        self.automl_config = configs["automl"]
    def eval(self, y_test, y_pred):
        """Calculate metrics RMSE, MAE, MPE, MAPE"""
        logger.info("Start evaluation")
        rmse_loss = mean_squared_error(y_test, y_pred, squared=False) # squared = False -> rmse
        mae_loss = mean_absolute_error(y_test, y_pred)
        mape_loss = mean_absolute_percentage_error(y_test, y_pred)
        
        return {"RMSE": rmse_loss,
                "MAE": mae_loss,
                "MAPE": mape_loss}
    
    def save_record(self, recorder, save_record):
        df = defaultdict(list)
        for iter, info in recorder.items():
            df["iterations"].append(iter)
            df["best_pipeline"].append(info[0])
            df["test_scores"].append(info[1])
            
        df = pd.DataFrame(df)
        df.to_csv(save_record, index=False)
    def preprocess_data(self, data):
        return data # dont need to preprocess so just return the original

    def run(self, dataset_path, save_record=None):
        # init recorer for saving results
        recorder = {}

        # Start
        df = pd.read_csv(dataset_path)
        pycaret.setup(data=df, target=self.target_field[0], train_size=self.train_size, silent=True, log_experiment=False, fold_shuffle=True);
        _X, _Y = self.setup_data(dataset_path)
        _X, _Y = self.preprocess_data((_X, _Y))
        X, Y = pd.DataFrame(_X), pd.DataFrame(_Y)
        all_models = pycaret.models()
        X_train, X_test, y_train, y_test = self.split_dataset(X, Y)
        pycaret.set_config("X_train", X_train)
        pycaret.set_config("y_train", y_train)
        pycaret.set_config("X_test", X_test)
        pycaret.set_config("y_test", y_test)
        # self.preprocess_data((X, y)) 
        for iter in range(self.n):
            logger.info(f"Run iteration {iter}")
        
            estimator = pycaret.compare_models(verbose=False, sort=self.automl_config["compare_model_sort"])
            # get id of top estimators
            id = all_models.loc[all_models["Reference"].str.endswith(estimator.__class__.__name__)].index[0]
            # create models for top estimators
            model = pycaret.create_model(id, verbose=False)
            # tune
            best_pipeline = pycaret.tune_model(model, optimize=self.automl_config["optimize"], verbose=False)

            # run infer on testset
            predicts = pycaret.predict_model(best_pipeline, verbose=False)
            
            # run evaluation
            y_pred = predicts["Label"]
            metric_scores = self.eval(y_test, y_pred)
            # logger.info(f"{metric_scores}")
            
            # save results
            recorder[iter] = [best_pipeline, metric_scores]
            all_metrics = pycaret.get_metrics()
            print(all_metrics)
            print(best_pipeline)
        
        if(save_record):
            self.save_record(recorder, save_record)
