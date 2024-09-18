from Hate_Speech_Classification.config.configuration import ConfigurationManager
from Hate_Speech_Classification.components.data_transformation import DataTransformation
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.Exception import CustomException
import sys


STAGE_NAME= "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):  # initializing empty constructor
        pass


    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_Transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        return data_transformation_artifact

# For integrating DVC we will use in this way
if __name__ =='__main__':
    try:
        logging.info(f"Stage {STAGE_NAME} Started")
        obj=DataTransformationTrainingPipeline()  # Calling the class
        obj.main()                           # Calling the main method
        logging.info(f" Stage {STAGE_NAME} Completed")

    except Exception as e:
        raise CustomException(e,sys) from e

# call the pipeline in main.py