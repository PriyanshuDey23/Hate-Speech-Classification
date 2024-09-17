from dataclasses import dataclass
from pathlib import Path

# Config.yaml
@dataclass(frozen=True)
class DataIngestionConfig:  # It is not an actual class , but a data class
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path