from src.mlproject.logger import logging
from src.mlproject.exceptions import CustomException
import sys

if __name__ == '__main__':
    logging.info('Test execution started')

    try:
        a = 1 / 0  # This will raise ZeroDivisionError
    except Exception as e:  # Catch the original exception
        logging.error('An exception occurred', exc_info=True)  # Optional: Logs traceback
        raise CustomException(e, sys)  # Wrap it in CustomException
