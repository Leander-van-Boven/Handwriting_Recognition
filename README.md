[Root](./README.md)
# Handwriting Recognition 2022 Group project
Group 6.

## Recognizers for Tasks 1 & 2 and Task 3
This repository contains both the recognizers for the DSS dataset, as well as for the IAM dataset. Note that, for 
running the bare recognizers, you **do not** need to clone the submodules listed in `./data`.
- Refer to the [IAM directory readme](./iam/readme.md) (`./iam`) for instructions on how to run the IAM recognizer.
- Refer to the [DSS directory readme](./dss/readme.md) (`./dss`) for instructions on how to run the DSS recognizer.

## Full installation
If you want to have full interaction with the CLI, to for instance run the recognizers on the full data-sets, you need
to clone all submodules. You can do so by running:
```shell
./install_full.sh
```
Note that this will take some time. This will set up a `pipenv` environment, as well as clone all submodules required
for this project.
## Command-line interface
```shell
$ py main.py --help
usage: main.py [-h] [--debug] [-c PATH] [-o PATH] [-s] [-l] {dss,iam} ...

positional arguments:
  {dss,iam}             the task to scope to: dss=Dead Sea Scrolls tasks; iam=IAM-dataset task

optional arguments:
  -h, --help            show this help message and exit
  --debug               print the config and parsed arguments
  -c PATH, --config PATH
                        the location of the config yaml file
  -o PATH, --outdir PATH
                        the directory to store the generated output files
  -s, --save-intermediate
                        save the output of all intermediate steps to disk
  -l, --load-intermediate
                        load all intermediate steps from disk, when available
```

```shell
$ py main.py dss --help
usage: main.py dss [-h] [-i PATH] [-S OPT] [-g GLOB]

optional arguments:
  -h, --help            show this help message and exit
  -i PATH, --indir PATH
                        the directory from which to load the input dataset
  -S OPT, --stage OPT   the stage to execute: line_segment, word_segment, segment_statistics, classify, ctc, write_output, full
  -g GLOB, --glob GLOB  the file pattern used to determine the images used as input
```
Note that the IAM task has identical command-line arguments, just with different defaults and with different available 
stages. By default, running `python3 main.py dss` or `python3 main.py iam` will let the full pipeline run on the full 
DSS or respectively IAM dataset, without saving or loading intermediate steps.