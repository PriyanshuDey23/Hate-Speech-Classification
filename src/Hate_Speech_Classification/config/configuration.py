from Hate_Speech_Classification.constrants import * # Import Everything
from Hate_Speech_Classification.utils.common import read_yaml,create_directories 
from Hate_Speech_Classification.entity.config_entity import *

class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,  # Return Box Type  # Ctrl+click to check the file path
            params_filepath=PARAMS_FILE_PATH):

            self.config=read_yaml(config_filepath)
            self.params=read_yaml(params_filepath)

            # From common.py
            create_directories([self.config.artifacts_root]) # I can call using the key name using Box Type

# Data ingestion

    def get_data_ingestion_config(self) -> DataIngestionConfig: # calling Data ingest config
        config=self.config.data_ingestion  # Storing the config

        create_directories([config.root_dir]) # Check config 

        # Define the custom return type of the function, check config, storing it
        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir    
        )

        return data_ingestion_config
    
# Data Validation
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config  
    
# Data Transformation

    def get_data_Transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir]) 

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            Transformed_filename=config.Transformed_filename,
            Data_dir=config.Data_dir,
            Id=config.Id,
            Axis=config.Axis,
            Inplace=config.Inplace,
            Drop_Columns=config.Drop_Columns,
            Class=config.Class,
            Label=config.Label,
            Tweet=config.Tweet


        )

        return data_transformation_config

