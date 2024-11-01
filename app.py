from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion
#from src.mlproject.components.data_ingestion import DataIngestionConfig

from src.mlproject.components.data_transformation import DataTransformation, DataTransformationConfig

#from src.mlproject.components.data_transformation import DataTransformationConfig

if __name__ == '__main__':
    logging.info('Test execution started')

    try:
        data_ingestion=DataIngestion()
        #data_ingestion_config=DataTransformationConfig()
        #data_ingestion.initiate_data_ingestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig

        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    except Exception as e:  # Catch the original exception
        logging.error('An exception occurred', exc_info=True)  # Optional: Logs traceback
        raise CustomException(e, sys)  # Wrap it in CustomException
