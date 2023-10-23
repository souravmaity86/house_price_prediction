import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging, project_dir
from src.utils import load_object


class PredictPipeline:
    
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            # get file path
            preprocessor_file_path = os.path.join(
                project_dir, "artifacts", "preprocessor.pkl")
            model_file_path = os.path.join(
                project_dir, "artifacts", "model.pkl")
            
            # create object
            preprocessor = load_object(preprocessor_file_path)
            model = load_object(model_file_path)
            
            # transformed data
            data_scaled = preprocessor.transform(features)
            
            # prediction
            y_pred = model.predict(data_scaled)
            
            return y_pred
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e, sys)


class CustomData:
    
    def __init__(self, **kwargs):
        self.crim = kwargs['crim']
        self.zn = kwargs['zn']
        self.indus = kwargs['indus']
        self.chas = kwargs['chas']
        self.nox = kwargs['nox']
        self.rm = kwargs['rm']
        self.age = kwargs['age']
        self.dis = kwargs['dis']
        self.rad = kwargs['rad']
        self.tax = kwargs['tax']
        self.ptratio = kwargs['ptratio']
        self.b = kwargs['b']
        self.lstat = kwargs['lstat']
        
    def get_user_inputs(self):
        try:
            user_data_dict = {
                'crim': [self.crim],
                'zn': [self.zn],
                'indus': [self.indus],
                'chas': [self.chas],
                'nox': [self.nox],
                'rm': [self.rm],
                'age': [self.age],
                'dis': [self.dis],
                'rad': [self.rad],
                'tax': [self.tax],
                'ptratio': [self.ptratio],
                'b': [self.b],
                'lstat': [self.lstat]
            }
            # create a df
            df = pd.DataFrame(user_data_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e, sys)
