import os
import sys
from dataclasses import dataclass

from numpy.ma.core import absolute
#from catboost import CatBoostRegressor ***this broke idk****
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import  XGBRegressor

from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
from src.mlproject.utils import save_object,evaluate_model
from template import file_path


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_obj_file_path):
        try:

            logging.info('Split train and test input data')
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],train_arr[:,-1],
                                           test_arr[:,:-1],test_arr[:,-1])

            models={
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
            }

            params={
                'DecisionTreeRegressor': {
                    'criterion': ['gini', 'entropy','squared_error','absolute_error'],
                    'splitter': ['best', 'random'],
                    'max_features': ['auto', 'sqrt'],
                },
                'RandomForestRegressor': {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['auto', 'sqrt'],
                    'bootstrap': [True, False],
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'subsample': [0.6, 0.8, 1.0],
                },
                'KNeighborsRegressor': {
                    'n_neighbors': [1, 2, 3, 4, 5],
                    'weights': ['uniform', 'distance']
                },
                'LinearRegression': {
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'subsample': [0.6, 0.8, 1.0],
                }
            }
            #to see the best models
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params)
            best_model_score=max(sorted(model_report.values()))

            #to get best model name from dict

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model= models[best_model_name]

            #THRESHOLD

            if best_model_score<0.6:
                raise CustomException('The best model is less than 0.6, No best models')
            logging.info('Best model found on both training and test set')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_sc = r2_score(y_test,predicted)
            return r2_sc



        except Exception as e:
            raise CustomException(e,sys)

