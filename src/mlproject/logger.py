import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "LOGS"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join("LOGS", LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Write logs to a file
        logging.StreamHandler()           # Print logs to the console
    ]
)

# Use the logger instance
logger = logging.getLogger()
