from loguru import logger
from collections import defaultdict
import pandas as pd
from .base import BaseAutoML
import pycaret.regression as pycaret
from sklearn.metrics import (mean_squared_error, mean_absolute_error)
from utils.mape import mean_absolute_percentage_error
import numpy as np

            
class PycaretAutoML(BaseAutoML):
    def __init__(self, configs):
        super(PycaretAutoML, self).__init__(configs["num_iteration"], configs["train_size"])

        automl_config = configs["automl"]
    def setup_data(self, dataset_path):
        """Prepare dataset for training and evaluating
        """
        logger.info(f"Load dataset from {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        fields = self.fields + self.target_field
        df = df.filter(items=(self.fields + self.target_field), axis=1);
        
        # drop nan value in target var
        df.dropna(subset=self.target_field, inplace=True)

        pycaret.setup(data=df, target=self.target_field[0], train_size=self.train_size);
        
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
    
    def run(self, dataset_path, save_record=None):
        # init recorer for saving results
        recorder = {}
        
        # Start
        self.setup_data(dataset_path)
        # self.preprocess_data((X, y)) 
        for iter in range(self.n):
            logger.info(f"Run iteration {iter}")
            
            best_pipeline = pycaret.compare_models(verbose=False)
            
            # run infer on testset
            predicts = pycaret.predict_model(best_pipeline, verbose=False)
            
            # run evaluation
            y_test, y_pred = predicts[self.target_field], predicts["Label"]
            metric_scores = self.eval(y_test, y_pred)
            logger.info(f"{metric_scores}")
            
            # save results
            recorder[iter] = [best_pipeline, metric_scores]
        
        if(save_record):
            self.save_record(recorder, save_record)
