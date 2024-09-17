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
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: Path