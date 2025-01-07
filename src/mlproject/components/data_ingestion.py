import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.mlproject.exceptions import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import read_sql_data
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Step 1: Fetch data from SQL database
            logging.info("Reading data from SQL database...")
            df = read_sql_data()
            logging.info("Data successfully fetched from SQL database.")

            # Step 2: Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}.")

            # Step 3: Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Step 4: Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test datasets saved successfully.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error during data ingestion.")
            raise CustomException(e, sys)


def run_pipeline():
    logging.info("Pipeline execution started")

    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion complete. Train path: {train_path}, Test path: {test_path}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info(f"Data transformation complete. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Model training complete. R2 Score: {r2_score}")

        return {
            "status": "success",
            "r2_score": r2_score,
            "preprocessor_path": preprocessor_path
        }

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        result = run_pipeline()
        print(f"Pipeline completed successfully: {result}")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
