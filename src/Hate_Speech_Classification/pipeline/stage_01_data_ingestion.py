from Hate_Speech_Classification.config.configuration import ConfigurationManager
from Hate_Speech_Classification.components.data_ingestion import DataIngestion
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.Exception import CustomException
import sys



STAGE_NAME= "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):  # initializing empty constructor
        pass


    def main(self):
        config=ConfigurationManager()
        data_ingestin_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestin_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


# For integrating DVC we will use in this way
if __name__ =='__main__':
    try:
        logging.info(f"Stage {STAGE_NAME} Started")
        obj=DataIngestionTrainingPipeline()  # Calling the class
        obj.main()                           # Calling the main method
        logging.info(f" Stage {STAGE_NAME} Completed")

    except Exception as e:
        raise CustomException(e,sys) from e

# call the pipeline in main.py