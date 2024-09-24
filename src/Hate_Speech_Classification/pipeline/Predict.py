import os
import io
import sys
import keras
import pickle
from PIL import Image
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.constrants import *
from Hate_Speech_Classification.Exception import CustomException
from keras.utils import pad_sequences
from Hate_Speech_Classification.config.configuration import ConfigurationManager
from Hate_Speech_Classification.entity.config_entity import DataTransformationConfig
from Hate_Speech_Classification.entity.config_entity import ModelEvaluationConfig
from Hate_Speech_Classification.components.data_transformation import DataTransformation


class PredictionPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()  
        self.config = config_manager.get_model_evaluation_config()
        self.data_transformation = DataTransformation(config=DataTransformationConfig)
        self.model_path = self.config.model_path
        self.tokenizer_path = os.path.join(self.config.tokenizer_path, 'tokenizer.pickle') 
        
        
       
        
        


    def predict(self,words):
        logging.info("Running the predict function")
        try:
            load_model = keras.models.load_model(self.model_path)
            # tokenizer_path = os.path.join(self.config['tokenizer_path'], 'tokenizer.pickle')
            
            with open(self.tokenizer_path, 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            cleaned_text=self.data_transformation.concat_data_cleaning(words)
            cleaned_text = [cleaned_text]            
            print(cleaned_text)

            seq = load_tokenizer.texts_to_sequences(cleaned_text)
            padded = pad_sequences(seq, maxlen=300)
            print(seq)

            pred = load_model.predict(padded)
            pred
            
            print("pred", pred)
            if pred[0] > 0.5:
                return "hate and abusive"
            else:
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def run_pipeline(self,words):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            
            predicted_text=self.predict(words)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text

        except Exception as e:
            raise CustomException(e, sys) from e


    

