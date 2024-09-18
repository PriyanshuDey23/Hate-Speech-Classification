import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from Hate_Speech_Classification.Logging import logging 
from Hate_Speech_Classification.Exception import CustomException
from Hate_Speech_Classification.config.configuration import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config



    def imbalance_data_cleaning(self):

        try:
            logging.info("Entered into the imbalance_data_cleaning function")
            imbalance_data=pd.read_csv("artifacts\data_ingestion\imbalanced_data.csv")
            imbalance_data.drop(self.config.Id,axis=self.config.Axis , inplace = self.config.Inplace)
            logging.info(f"Exited the imbalance data_cleaning function and returned imbalance data {imbalance_data}")
            return imbalance_data 
        except Exception as e:
            raise CustomException(e,sys) from e 



    def raw_data_cleaning(self):

        try:
            logging.info("Entered into the raw_data_cleaning function")
            raw_data = pd.read_csv("artifacts/data_ingestion/raw_data.csv")
            raw_data.drop(self.config.Drop_Columns,axis = self.config.Axis,
            inplace = self.config.Inplace)


            raw_data[raw_data[self.config.Class]==0][self.config.Class]=1

            # replace the value of 0 to 1
            raw_data[self.config.Class].replace({0:1},inplace=True)

            # Let's replace the value of 2 to 0.
            raw_data[self.config.Class].replace({2:0}, inplace = True)

            # Let's change the name of the 'class' to label
            raw_data.rename(columns={self.config.Class:self.config.Label},inplace =True)
            logging.info(f"Exited the raw_data_cleaning function and returned the raw_data {raw_data}")
            return raw_data

        except Exception as e:
            raise CustomException(e,sys) from e



    def concat_dataframe(self):

        try:
            logging.info("Entered into the concat_dataframe function")
            # Let's concatinate both the data into a single data frame.
            frame = [self.raw_data_cleaning(), self.imbalance_data_cleaning()]
            df = pd.concat(frame)
            print(df.head())
            logging.info(f"returned the concatinated dataframe {df}")
            return df

        except Exception as e:
            raise CustomException(e, sys) from e



    def concat_data_cleaning(self, words):

        try:
            logging.info("Entered into the concat_data_cleaning function")
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)

            # Remove stopwords and apply stemming
            words = [word for word in words.split() if word not in stopword]
            words = " ".join([stemmer.stem(word) for word in words])

            logging.info("Exited the concat_data_cleaning function")
            return words
        except Exception as e:
            raise CustomException(e, sys) from e



    def initiate_data_transformation(self):
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            
            # Cleaning and transforming data
            df = self.concat_dataframe()
            df[self.config.Tweet] = df[self.config.Tweet].apply(self.concat_data_cleaning)

            # Save the transformed data
            os.makedirs(self.config.root_dir, exist_ok=True)
            df.to_csv(self.config.Transformed_filename, index=False, header=True)

            # Return the DataTransformationConfig object
            data_transformation_artifact = DataTransformationConfig(
                root_dir=self.config.root_dir,
                Transformed_filename=self.config.Transformed_filename,
                Data_dir=self.config.Data_dir,
                Id=self.config.Id,
                Axis=self.config.Axis,
                Inplace=self.config.Inplace,
                Drop_Columns=self.config.Drop_Columns,
                Class=self.config.Class,
                Label=self.config.Label,
                Tweet=self.config.Tweet
            )
            logging.info("Returning the DataTransformationArtifacts")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e