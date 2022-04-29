# Handwriting Recognition 2022 Group project
Group 8.

## Preliminaries
We assume a Linux environment with Python (3.8), Pipenv and Git installed.

## Installation
```shell
./install.sh
```
This will initiate and download all submodules that contain all dataset. You have to make sure that you have access to 
all restricted repositories (containing DSS data) before running this command.

## Running
In JetBrains PyCharm, run the "DSS Test" configuration.

## To do, pipeline-specific
- [ ] Figure out what is specified in argparse, what in config.yaml, whether we need anything else
- [ ] Add abstract class for classifiers and segmenters, specifying tasks like `train`, `test`, etc.
- [ ] Figure out how to work with end-to-end solutions, if we will work with one
- [ ] IAM implementation, will it differ from DSS?