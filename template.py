import os
from pathlib import Path  
import logging

# Creating Logging string
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')# level :- informtion related to log # current time- asctime, along with the message

project_name="Hate_Speech_Classification" # As Per Requirement

# List of Files and Folder need to be created

list_of_files=[
    ".github/workflows/.gitkeep", # For getting reflected in the git hub
    f"src/{project_name}/__init__.py", # we are installing it as a local package, so we need a constructor(__init__.py)
    f"src/{project_name}/components/__init__.py", # Where ever it is present it will be considered as a package
    f"src/{project_name}/Logging/__init__.py",
    f"src/{project_name}/Exception/__init__.py"
    f"src/{project_name}/utils/__init__.py", # General Struture can be defined here
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py", # Contain All the configuration
    f"src/{project_name}/pipeline/__init__.py", # Pipeline Folder
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constrants/__init__.py",
    "config/config.yaml",
    "dvc.yaml", # Mlops tool :- DVC
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb", # notebook experiments
    "templates/index.html"

]

for filepath in list_of_files:
    filepath= Path(filepath) # define the path type, it will ignore the(/) and create the files and folder
    filedir,filename=os.path.split(filepath)# For Separating the file and Folder name

    # Create the directory
    if filedir != "":       # Not  empty,present(if folder is present)
        os.makedirs(filedir,exist_ok=True) # Make directory, exist_ok=True:- If it is created then it won't be created again
        logging.info(f" Creating Directory ; {filedir} for the file: {filename}") 

    # Creating the files
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): # Check  file not exists or if the size of the file is zero  ,if it is not empty then it will not replace it

        with open(filepath, "w" ) as f: # Create that file
            pass
            logging.info(f"Creating Empty File: {filepath}")


    else:
        logging.info(f"{filename} is already exists ")

# All Files And Folder will be created