"""import os
import sys
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.exceptions import CustomException
from src.mlproject.logger import logging


def run_pipeline():
    try:
        logging.info("Starting the training pipeline...")

        # Step 1: Data Transformation
        logging.info("Starting data transformation...")
        data_transformation = DataTransformation()

        train_path = os.path.join("artifact", "train.csv")
        test_path = os.path.join("artifact", "test.csv")

        transformation_result = data_transformation.initiate_data_transformation(train_path, test_path)
        train_arr, test_arr, preprocessor_path = transformation_result

        logging.info(f"Data transformation complete. Preprocessor saved at {preprocessor_path}")

        # Step 2: Model Training
        logging.info("Starting model training...")
        model_trainer = ModelTrainer()

        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)

        logging.info(f"Model training complete. R2 score: {r2_score}")

        logging.info("Training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during the pipeline execution: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()
    """
