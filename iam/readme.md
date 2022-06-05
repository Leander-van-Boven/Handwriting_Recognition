# Handwriting Recognition 2022 Task 3: IAM-dataset
## How to run
We assume a Linux-environment, with python 3.8 installed. You may need to add execution privileges for the shell
files, by running ```chmod +x *.sh```.
- To install all dependencies, run `./install.sh`. This will install `pipenv` if it isn't installed yet, and install the
  `pipenv` environment with all required dependencies.
  > Note that you don't have to run the installer if you have already done so for the DSS-recognizer. The recognizers
  > share the same `pipenv` environment.
- To run the recognizer as per the specified requirements, run `./run.sh /path/to/test/images`. This will run the 
  recognizer, and save the resulting text files in a folder `./results`.