import os
import sys
from dataclasses import dataclass

#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.exceptions import CustomException

from src.mlproject.logger import logging

from src.mlproject.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Split training and test input data
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models and hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models and retrieve report + fitted models
            model_report, fitted_models = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get the best model name and score
            best_model_name = max(model_report, key=model_report.get)  # Model with highest score
            best_model_score = model_report[best_model_name]
            best_model = fitted_models[best_model_name]  # Retrieve the fitted model

            # Check if the best model meets the threshold
            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found. Best model score below threshold.")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            # Make predictions using the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


"""import os
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




            if best_model_score < 0.6:
                logging.warning("Best model score is below threshold. Proceeding with the best available model.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Saving model to {self.model_trainer_config.trained_model_file_path}")

            predicted=best_model.predict(X_test)

            r2_sc = r2_score(y_test,predicted)
            return r2_sc



        except Exception as e:
            raise CustomException(e,sys)
            """




