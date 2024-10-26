import os
import sys
from src.mlproject.exceptions import CustomException
from src.mlproject.logger import logging
import pandas as pd
from src.mlproject.utils import read_sql_data
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass

class DataIngestionConfig:
    train_data_path: str= os.path.join('artifact', 'train.csv')
    test_data_path: str= os.path.join('artifact', 'test.csv')
    raw_data_path: str= os.path.join('artifact', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            #reading mysql data
            logging.info('Reading from mysql database')

            #from utils
            df=read_sql_data()


            #making dir
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # conv df to csv for storage, raw
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            #spilitting test train


            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)


            # conv df to csv for storage, train and test
            df.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('data ingestion completed')

            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


