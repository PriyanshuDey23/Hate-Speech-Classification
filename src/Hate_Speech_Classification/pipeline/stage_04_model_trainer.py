from Hate_Speech_Classification.config.configuration import ConfigurationManager
from Hate_Speech_Classification.components.model_trainer import ModelTrainer
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.Exception import CustomException
import sys


STAGE_NAME= "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):  # initializing empty constructor
        pass


    def main(self):
        config=ConfigurationManager()  
        model_trainer_config=config.get_model_trainer_config()
        model_trainer=ModelTrainer(config=model_trainer_config)
        model_trainer=model_trainer.initiate_model_trainer()



# For integrating DVC we will use in this way
if __name__ =='__main__':
    try:
        logging.info(f"Stage {STAGE_NAME} Started")
        obj=ModelTrainerTrainingPipeline()  # Calling the class
        obj.main()                           # Calling the main method
        logging.info(f" Stage {STAGE_NAME} Completed")

    except Exception as e:
        raise CustomException(e,sys) from e

# call the pipeline in main.py