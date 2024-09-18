from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.Exception import CustomException
import sys


# Call the pipeline  (src\Hate_Speech_Classification\pipeline\stage_01_data_ingestion.py) in main .py 

from Hate_Speech_Classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME="Data Ingestion stage"
try:
    logging.info(f"Stage {STAGE_NAME} Started")
    obj=DataIngestionTrainingPipeline()  # Calling the class
    obj.main()                           # Calling the main method,
    logging.info(f" Stage {STAGE_NAME} Completed")

except Exception as e:
    raise CustomException(e,sys) from e

# After executing , Artifacts folder , zip file download , unzip of the file will happen




# Call the pipeline  (src\Hate_Speech_Classification\pipeline\stage_02_data_validation.py) in main .py 

from Hate_Speech_Classification.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

STAGE_NAME="Data Validation Stage"
try:
    logging.info(f"Stage {STAGE_NAME} Started")
    obj=DataValidationTrainingPipeline()  # Calling the class
    obj.main()                           # Calling the main method,
    logging.info(f" Stage {STAGE_NAME} Completed")

except Exception as e:
    raise CustomException(e,sys) from e

# After executing , artifacts\data_validation\status.txt will get created




# Call the pipeline  (src\Hate_Speech_Classification\pipeline\stage_03_data_transformation.py) in main .py 

from Hate_Speech_Classification.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME="Data Transformation Stage"
try:
    logging.info(f"Stage {STAGE_NAME} Started")
    obj=DataTransformationTrainingPipeline()  # Calling the class
    obj.main()                           # Calling the main method,
    logging.info(f" Stage {STAGE_NAME} Completed")

except Exception as e:
    raise CustomException(e,sys) from e

# After executing , artifacts\data_transformation\final.csv will get created


