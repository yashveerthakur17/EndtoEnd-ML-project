from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion
#from src.mlproject.components.data_ingestion import DataIngestionConfig


if __name__ == '__main__':
    logging.info('Test execution started')

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:  # Catch the original exception
        logging.error('An exception occurred', exc_info=True)  # Optional: Logs traceback
        raise CustomException(e, sys)  # Wrap it in CustomException
