from dataclasses import dataclass
from pathlib import Path

# Data Ingestion
@dataclass(frozen=True)
class DataIngestionConfig:  # It is not an actual class , but a data class
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path

# Data Validation

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: Path

# Data Transformation

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    Transformed_filename: Path
    Data_dir: str
    Id: str
    Axis: int
    Inplace: bool
    Drop_Columns: list
    Class: str
    Label: str
    Tweet: str


# Model Trainer


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    x_test_data_path: Path
    x_train_data_path: Path
    y_test_data_path: Path
    Random_state: int
    Epoch: int
    Batch_size: int
    Validation_Split: float
    Max_Words: int
    Max_Len: int
    Loss: str
    Metrics: list
    Activation: str
    test_size: float
    layers: int