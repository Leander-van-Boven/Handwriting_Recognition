#!/bin/bash

DIR=$(realpath "$1")
cd "${BASH_SOURCE%/*}/.." || exit
pipenv run python3 main.py --save-intermediate --outdir ./iam iam --indir "$DIR" --glob '*'