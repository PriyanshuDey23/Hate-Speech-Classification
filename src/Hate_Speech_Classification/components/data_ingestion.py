import os
import urllib.request as request  # Download the data set from url
import zipfile # Unzip the data
from Hate_Speech_Classification.Logging import logging # Logging
from Hate_Speech_Classification.utils.common import get_size # Get datasize
from Hate_Speech_Classification.config.configuration import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self,config: DataIngestionConfig): # we will get get data ingestion config
        self.config=config

    # Download File Method
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):  # Check the directory,if not exists
            filename,headers=request.urlretrieve(           # UrlRetrive:-Retrieve a URL into a temporary location on disk.
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logging.info(f"{filename} download! with the following info: \n{headers}") # Header=url
        else:
            logging.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")


    # Extract the zip file
    def extract_zip_file(self):
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path) # Extract all file here (unzip_path)
