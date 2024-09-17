import os

from Hate_Speech_Classification.Exception import CustomException
import sys

from Hate_Speech_Classification.config.configuration import DataValidationConfig




class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    
    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts","data_ingestion")) # Check whether this files are availabke or not
            
            
            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:    # If It is not available
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise CustomException(e,sys) from e
