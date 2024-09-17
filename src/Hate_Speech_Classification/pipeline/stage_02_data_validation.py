from Hate_Speech_Classification.config.configuration import ConfigurationManager
from Hate_Speech_Classification.components.data_validation import DataValidation
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.Exception import CustomException
import sys


STAGE_NAME= "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):  # initializing empty constructor
        pass


    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exist()


# For integrating DVC we will use in this way
if __name__ =='__main__':
    try:
        logging.info(f"Stage {STAGE_NAME} Started")
        obj=DataValidationTrainingPipeline()  # Calling the class
        obj.main()                           # Calling the main method
        logging.info(f" Stage {STAGE_NAME} Completed")

    except Exception as e:
        raise CustomException(e,sys) from e

# call the pipeline in main.py