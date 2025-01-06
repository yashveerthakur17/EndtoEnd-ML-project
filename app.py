from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion
#from src.mlproject.components.data_ingestion import DataIngestionConfig
import numpy as np

from src.mlproject.components.data_transformation import DataTransformation, DataTransformationConfig

#from src.mlproject.components.data_transformation import DataTransformationConfig

if __name__ == '__main__':
    logging.info('Test execution started')

    """try:
        data_ingestion=DataIngestion()
        #data_ingestion_config=DataTransformationConfig()
        #data_ingestion.initiate_data_ingestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig

        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Model training completed with R2 score: {r2_score}")

    except Exception as e:  # Catch the original exception
        logging.error('An exception occurred', exc_info=True)  # Optional: Logs traceback
        raise CustomException(e, sys)  # Wrap it in CustomException"""

    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion complete. Train path: {train_path}, Test path: {test_path}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_path,
                                                                                                      test_path)
        logging.info(f"Data transformation complete. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Model training complete. R2 Score: {r2_score}")

    except Exception as e:
        raise CustomException(e, sys)
