import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.constrants import * # Import Everything
from Hate_Speech_Classification.Exception import CustomException
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix
from Hate_Speech_Classification.config.configuration import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def load_tokenizer(self):
        """
        Load the tokenizer from a file (tokenizer.pickle).
        """
        try:
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            logging.info("Tokenizer loaded successfully.")
            return tokenizer
        except FileNotFoundError as e:
            raise CustomException(f"Tokenizer file not found: {str(e)}", sys)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def save_tokenizer(self, tokenizer):
        try:
            tokenizer_file_path = os.path.join(self.config.tokenizer_path, 'tokenizer.pickle')
            with open(tokenizer_file_path, 'wb') as handle:
                pickle.dump(tokenizer, handle)
            logging.info(f"Tokenizer saved successfully at {tokenizer_file_path}.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def save_accuracy(self, accuracy):
        try:
            accuracy_file_path = os.path.join(self.config.metric_file_name, 'accuracy.txt')
            with open(accuracy_file_path, 'w') as file:
                file.write(f"Test Accuracy: {accuracy}\n")
            logging.info(f"Accuracy saved successfully at {accuracy_file_path}.")
        except Exception as e:
            raise CustomException(e, sys) from e
    


    def evaluate(self):
        
        try:
            logging.info("Entering into to the evaluate function of Model Evaluation class")
            print(self.config.x_test_data_path)

            # Loading test data

            x_test = pd.read_csv(self.config.x_test_data_path,index_col=0).squeeze()
            print(x_test)
            y_test = pd.read_csv(self.config.x_test_data_path,index_col=0).squeeze()
            print(y_test)


            # Check for NaN values
            if x_test.isnull().any():
                logging.warning("x_test contains NaN values, filling with empty string.")
                x_test = x_test.fillna('')
            if y_test.isnull().any():
                logging.warning("y_test contains NaN values, filling with 0.")
                y_test = pd.to_numeric(y_test, errors='coerce')  # Convert to numeric, coercing errors to NaN
                y_test = y_test.fillna(0)  # Or handle NaNs as needed
                logging.info(f"Unique values in y_test after conversion: {y_test.unique()}")

            # Preprocess test data
            x_test = x_test.astype(str)
            y_test = y_test.astype(float)

            
            
            # Load the tokenizer
            tokenizer = self.load_tokenizer()

            # Load trained model
            load_model=keras.models.load_model(self.config.model_path)

            


            


            # Tokenizing the test data
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences,maxlen=self.config.Max_Len)
            


            # Evaluate model

            accuracy = load_model.evaluate(test_sequences_matrix,y_test)
            logging.info(f"the test accuracy is {accuracy}")


            # Save tokenizer and accuracy
            self.save_tokenizer(tokenizer)
            self.save_accuracy(accuracy)

             # Making predictions

            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] >= 0.5:
                    res.append(1)  # Positive class
                else:
                    res.append(0)  # Negative class

            
            # Confusion matrix
            print(confusion_matrix(y_test,res))
            logging.info(f"the confusion_matrix is {confusion_matrix(y_test,res)} ")
            return accuracy
        
        
        except Exception as e:
            raise CustomException(e, sys) from e



    def initiate_model_evaluation(self) -> dict:
        try:
            logging.info("Initiating Model Evaluation")

            # Evaluate the current model
            trained_model_accuracy = self.evaluate()
            
            # Log the accuracy for debugging
            logging.info(f"Trained Model Accuracy: {trained_model_accuracy}")

            # If trained_model_accuracy is a tuple or list, assume it returns (loss, accuracy)
            if isinstance(trained_model_accuracy, (list, tuple)):
                accuracy = trained_model_accuracy[1]  # Assuming accuracy is at index 1
            else:
                accuracy = trained_model_accuracy  # If it's just a single scalar value
            
            # Ensure accuracy is a float before comparison
            is_model_accepted = float(accuracy) >= 0.5  
            logging.info("Trained model accepted." if is_model_accepted else "Trained model not accepted.")

            return {
                'is_model_accepted': is_model_accepted,
                'accuracy': accuracy
            }

        except Exception as e:
            raise CustomException(e, sys) from e
        
        
  