from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.Exception import CustomException
import sys


# Call the pipeline  (src\Hate_Speech_Classification\pipeline\stage_01_data_ingestion.py) in main .py 

from Hate_Speech_Classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME="Data Ingestion stage"
try:
    logging.info(f"Stage {STAGE_NAME} Started")
    obj=DataIngestionTrainingPipeline()  # Calling the class
    obj.main()                           # Calling the main method,Start the data ingestion part
    logging.info(f" Stage {STAGE_NAME} Completed")

except Exception as e:
    raise CustomException(e,sys) from e

# After executing , Artifacts folder , zip file download , unzip of the file will happen