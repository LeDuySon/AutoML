from loguru import logger
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class BaseAutoML():
    def __init__(self, num_iteration=30, train_size=0.8):
        self.n = num_iteration
        self.train_size = train_size
        self.test_size = 1 - train_size
        # columns for training
        self.fields = ["vec", "deltachi", "delta", "deltahmix", "deltasmix"]
        # target columns
        self.target_field = ["ys"] 
        
    def setup_data(self, dataset_path):
        """Prepare dataset for training and evaluating

        Returns:
            X: Input data fields
            y: Output data field
        """
        logger.info(f"Load dataset from {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        # drop nan value in target var
        df.dropna(subset=self.target_field, inplace=True)
        
        X, y = df[self.fields], df[self.target_field]
        if(len(y.shape) == 2):
            y = np.ravel(y)
            
        assert X.shape[0] == y.shape[0], "Mismatch shape"
        
        logger.debug(f"\n{df.describe()}")
        logger.debug(f"Dataset shape {X.shape}")
        logger.debug(f"Target shape {y.shape}")
        return X, y
    
    def split_dataset(self, X, y):
        """Split dataset into training set, test set"""
        logger.info("Start split dataset into train and test set")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69,
                                                    train_size=self.train_size, test_size=self.test_size)
        
        logger.debug(f"X train: {X_train.shape}")
        logger.debug(f"y train: {y_train.shape}")
        logger.debug(f"X test: {X_test.shape}")
        logger.debug(f"y test: {y_test.shape}")
        
        return (X_train, X_test, y_train, y_test)
    
    def fit(self):
        """Fit train dataset into automl for training"""
        raise NotImplementedError()
    
    def infer(self):
        """Run inference on test set"""
        raise NotImplementedError()
    
    def preprocess_data(self):
        """Preprocess the data before training and evaluating"""
        raise NotImplementedError()
    
    def eval(self):
        """Do model evaluation"""
        raise NotImplementedError()
    
    def run(self):
        raise NotImplementedError()
    
if __name__ == "__main__":
    dataset_path = "/home/ailab-rgb-1080ti/Work/son-btl/cacvandehiendai/datasets/ys1a.csv"
    automl = BaseAutoML()
    
    X, y = automl.setup_data(dataset_path) 
    automl.split_dataset(X, y)