# Remain Common in every Project

import setuptools

# IMAGE Seeker Package in pypi

with open("README.md","r",encoding="utf-8") as f: # Open in write mode
    long_description=f.read()

__version__="0.0.0"

REPO_NAME= "Hate_Speech_Classification"
AUTHOR_USER_NAME="PriyanshuDey23"
SRC_REPO= "Hate-Speech-Classification"
AUTHOR_EMAIL="priyanshudey.ds@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="It is a NLP Problem Statement",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/PriyanshuDey23/Hate-Speech-Classification",
    project_urls={
        "Bug Tracker":f"https://github.com/PriyanshuDey23/Hate-Speech-Classification/issues"
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")

)