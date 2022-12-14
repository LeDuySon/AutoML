from loguru import logger
from collections import defaultdict
import pandas as pd
from tpot import TPOTRegressor
from .base import BaseAutoML
from sklearn.metrics import (mean_squared_error, mean_absolute_error)
from utils.mape import mean_absolute_percentage_error
                             
class TpotAutoML(BaseAutoML):
    def __init__(self, configs):
        super(TpotAutoML, self).__init__(configs["num_iteration"], configs["train_size"])

        self.automl_config = configs["automl"]
        self.num_fold = configs["num_cross_validation_fold"]
        
    def init_pipeline_optimizer(self, automl_config):
        pipeline_optimizer = TPOTRegressor(generations=automl_config["generations"], 
                                            population_size=automl_config["population_size"], 
                                            scoring="neg_mean_squared_error",
                                            cv=self.num_fold,
                                            n_jobs=-1,
                                            config_dict=automl_config["config_dict"],
                                            memory=automl_config["memory"],
                                            verbosity=automl_config["verbosity"])
        
        return pipeline_optimizer
    
    def fit(self, pipeline_optimizer, X_train, y_train):
        logger.info(f"Start fit on training set with kfold {self.num_fold}")
        pipeline_optimizer.fit(X_train, y_train)
        logger.info("Finish training")
        
        # return best pipeline
        return pipeline_optimizer.fitted_pipeline_
        
    def infer(self, pipeline_optimizer, X_test):
        logger.info("Start inference on test data")
        y_pred = pipeline_optimizer.predict(X_test)
        return y_pred
        
    def eval(self, y_test, y_pred):
        """Calculate metrics RMSE, MAE, MPE, MAPE"""
        logger.info("Start evaluation")
        rmse_loss = mean_squared_error(y_test, y_pred, squared=False) # squared = False -> rmse
        mae_loss = mean_absolute_error(y_test, y_pred)
        mape_loss = mean_absolute_percentage_error(y_test, y_pred)
        
        return {"RMSE": rmse_loss,
                "MAE": mae_loss,
                "MAPE": mape_loss}    
        
    def preprocess_data(self, data):
        return data # dont need to preprocess so just return the original\
    
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
            logger.info(f"Run iteration {iter}")
            # init pipeline optimizer
            pipeline_optimizer = self.init_pipeline_optimizer(self.automl_config)
            
            # split train/test dataset
            X_train, X_test, y_train, y_test = self.split_dataset(X, y)
            
            # training automl model, tpot already have kfold 
            best_pipeline = self.fit(pipeline_optimizer, X_train, y_train)
            
            # run infer on testset
            y_pred = self.infer(pipeline_optimizer, X_test)
            
            # run evaluation 
            metric_scores = self.eval(y_test, y_pred)
            logger.info(f"{metric_scores}")
            
            # save results
            recorder[iter] = [best_pipeline, metric_scores]
        
        if(save_record):
            self.save_record(recorder, save_record)
