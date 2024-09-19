import os 
import sys
import pickle
import pandas as pd
from Hate_Speech_Classification.Logging import logging
from Hate_Speech_Classification.constrants import * # Import Everything
from Hate_Speech_Classification.Exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from Hate_Speech_Classification.config.configuration import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
        


    def spliting_data(self):
        try:
            logging.info("Entered the spliting_data function")
            df = pd.read_csv("artifacts/data_transformation/final.csv", index_col=False)
            logging.info("Splitting the data into x and y")
            x = df["tweet"]
            y = df["label"]

            # Handle NaN and non-string values in the 'tweet' column
            logging.info("Checking and handling NaN or float values in the text data")
            x = x.fillna('')  # Replace NaN with empty strings
            x = x.apply(lambda text: str(text))  # Convert any float or other types to strings

            # Optional: Convert text to lowercase
            x = x.apply(lambda text: text.lower())

            logging.info("Applying train_test_split on the data")
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=self.config.test_size, random_state=self.config.Random_state
            )

            logging.info(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)



    def tokenizing(self, x_train):
        try:
            logging.info("Tokenizing the data")
            tokenizer = Tokenizer(num_words=self.config.Max_Words)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            sequences_matrix = pad_sequences(sequences, maxlen=self.config.Max_Len)
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys)
        

    

        


    def get_model(self):
        try:
            model = Sequential()
            model.add(Embedding(input_dim=self.config.Max_Words, output_dim=self.config.layers, input_length=self.config.Max_Len))
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(1, activation=self.config.Activation))
            model.summary()
            model.compile(loss=self.config.Loss, optimizer=RMSprop(), metrics=self.config.Metrics)
            return model
        except Exception as e:
            raise CustomException(e, sys)




    def initiate_model_trainer(self):
        try:
            logging.info("Initiating model training")
            x_train, x_test, y_train, y_test = self.spliting_data()

            model = self.get_model()

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info("Training the model")
            model.fit(sequences_matrix, y_train, batch_size=self.config.Batch_size, epochs=self.config.Epoch, validation_split=self.config.Validation_Split)

            logging.info("Saving tokenizer and model")
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.config.root_dir, exist_ok=True)
            model.save(self.config.trained_model_path)

            logging.info("Saving test and train data")
            x_test.to_csv(self.config.x_test_data_path)
            y_test.to_csv(self.config.y_test_data_path)
            x_train.to_csv(self.config.x_train_data_path)

            return {
                "trained_model_path": self.config.trained_model_path,
                "x_test_path": self.config.x_test_data_path,
                "y_test_path": self.config.y_test_data_path,
            }
        except Exception as e:
            raise CustomException(e, sys)