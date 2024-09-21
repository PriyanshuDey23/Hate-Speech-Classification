from Hate_Speech_Classification.config.configuration import ConfigurationManager
from Hate_Speech_Classification.components.model_evaluation import ModelEvaluation
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.Exception import CustomException
import sys



STAGE_NAME= "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):  # initializing empty constructor
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_eval_config = config_manager.get_model_evaluation_config()
        evaluation = ModelEvaluation(model_eval_config)
        evaluation.initiate_model_evaluation()



# For integrating DVC we will use in this way
if __name__ =='__main__':
    try:
        logging.info(f"Stage {STAGE_NAME} Started")
        obj=ModelEvaluationTrainingPipeline()  # Calling the class
        obj.main()                           # Calling the main method
        logging.info(f" Stage {STAGE_NAME} Completed")

    except Exception as e:
        raise CustomException(e,sys) from e

# call the pipeline in main.py